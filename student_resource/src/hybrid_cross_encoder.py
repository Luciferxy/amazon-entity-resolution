"""
Hybrid Cascade Cross-Encoder for Business Entity Resolution
===========================================================
Two-Stage Cascade Architecture:
  Stage 1 (LightGBM): Ultra-fast scoring on all candidate pairs.
    - Confident matches (p >= high_thr) -> accepted directly.
    - Confident negatives (p <= low_thr) -> discarded directly.
  Stage 2 (Cross-Encoder): Deep contextual transformer rescoring for
    only the ambiguous "borderline" candidate pairs (low_thr < p < high_thr).

Optimized for NVIDIA T4 GPU in Google Colab (fp16 batched inference).
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import W

# Default cross-encoder backbone (fast, accurate 6-layer transformer)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PairTextDataset(Dataset):
    """Dataset of text pairs for Cross-Encoder tokenization."""
    def __init__(self, texts_a, texts_b, tokenizer, max_length=128):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        return self.texts_a[idx], self.texts_b[idx]


def collate_pairs(batch, tokenizer, max_length=128):
    texts_a, texts_b = zip(*batch)
    return tokenizer(
        list(texts_a),
        list(texts_b),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


class FastCrossEncoder:
    """Lightweight Cross-Encoder inference engine with fp16 acceleration."""
    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Cross-Encoder '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Distributing Cross-Encoder inference across {torch.cuda.device_count()} GPUs (DataParallel)...")
            self.model = nn.DataParallel(self.model)

        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    @torch.inference_mode()
    def predict_probs(self, texts_a, texts_b, batch_size=256, max_length=128):
        """Predict match probabilities for text pairs."""
        dataset = PairTextDataset(texts_a, texts_b, self.tokenizer, max_length=max_length)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_pairs(b, self.tokenizer, max_length=max_length),
            num_workers=2 if self.device == "cuda" else 0,
            pin_memory=(self.device == "cuda"),
        )

        probs_all = []
        t0 = time.time()
        n_total = len(texts_a)

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Handle both single-output (regression/logit) and 2-class softmax
            if logits.shape[-1] == 1:
                p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            elif logits.shape[-1] == 2:
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            else:
                p = torch.sigmoid(logits[:, 0]).cpu().numpy()

            probs_all.append(p)
            if (step + 1) % 50 == 0:
                scored = min((step + 1) * batch_size, n_total)
                rate = scored / (time.time() - t0 + 1e-5)
                print(f"    [Cross-Encoder] {scored:,}/{n_total:,} pairs ({rate:.0f} pairs/sec)", flush=True)

        return np.concatenate(probs_all)


def cascade_rescore(
    candidates_df,
    lgb_probs,
    s1_df,
    pool_df,
    low_thr=0.15,
    high_thr=0.70,
    ce_weight=0.50,
    model_name=DEFAULT_MODEL,
    batch_size=256,
):
    """
    Apply Cascade Reranking:
      1. Identify borderline candidate pairs (low_thr < p_lgb < high_thr).
      2. Score ONLY borderline pairs with the Cross-Encoder.
      3. Blend scores: p_final = (1 - ce_weight) * p_lgb + ce_weight * p_ce.
    """
    t_start = time.time()
    p_final = np.array(lgb_probs, dtype=np.float32).copy()

    if model_name == DEFAULT_MODEL:
        finetuned_dir = W / "cross_encoder_finetuned"
        if finetuned_dir.exists():
            model_name = str(finetuned_dir)
            print(f"Using fine-tuned Cross-Encoder weights from: {model_name}")

    # Identify borderline pairs
    borderline_mask = (lgb_probs > low_thr) & (lgb_probs < high_thr)
    n_borderline = int(borderline_mask.sum())
    total_pairs = len(candidates_df)

    print(f"\n=======================================================")
    print(f" Hybrid Cascade Rescoring")
    print(f" Total Candidate Pairs : {total_pairs:,}")
    print(f" Confident Positives  : {(lgb_probs >= high_thr).sum():,} (kept as-is)")
    print(f" Confident Negatives  : {(lgb_probs <= low_thr).sum():,} (kept as-is)")
    print(f" Borderline Pairs     : {n_borderline:,} ({n_borderline/total_pairs:.1%}) -> Rescoring with Cross-Encoder")
    print(f"=======================================================")

    if n_borderline == 0:
        print("No borderline pairs found within threshold window. Returning LightGBM probabilities.")
        return p_final

    # Extract text strings for borderline pairs
    borderline_df = candidates_df[borderline_mask].reset_index(drop=True)
    
    s1_map = dict(zip(
        s1_df["entity_id"].to_numpy(dtype=object),
        (s1_df["name_full"] + " | " + s1_df["addr"]).to_numpy(dtype=object)
    ))
    pool_map = dict(zip(
        pool_df["entity_id"].to_numpy(dtype=object),
        (pool_df["name_full"] + " | " + pool_df["addr"]).to_numpy(dtype=object)
    ))

    texts_a = [s1_map.get(s, "") for s in borderline_df["s1"].to_numpy(dtype=object)]
    texts_b = [pool_map.get(c, "") for c in borderline_df["cand"].to_numpy(dtype=object)]

    # Run Cross-Encoder inference
    ce = FastCrossEncoder(model_name=model_name)
    ce_probs = ce.predict_probs(texts_a, texts_b, batch_size=batch_size)

    # Blend probabilities
    blended = (1.0 - ce_weight) * lgb_probs[borderline_mask] + ce_weight * ce_probs
    p_final[borderline_mask] = blended

    total_time = time.time() - t_start
    print(f"\n[DONE] Cascade rescoring completed in {total_time:.1f}s ({total_time/60:.1f} min)!")
    return p_final
