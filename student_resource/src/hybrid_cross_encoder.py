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


def full_cross_encoder_score(
    candidates_df,
    s1_df,
    pool_df,
    model_name=DEFAULT_MODEL,
    batch_size=512,
    chunk_size=500_000,
    cache_tag="test_ce_full",
):
    """
    Option 3: Full Cross-Encoder Scoring on ALL candidate pairs (e.g. 5.7M).

    Features:
      - Multi-GPU DataParallel inference (both T4 GPUs active).
      - FP16 automatic mixed precision.
      - 500k-pair chunked execution with per-chunk parquet caching (resumable).
      - Uses raw business_name + business_address + country for maximum
        semantic signal to the transformer.
    """
    from config import CACHE
    final_cache = CACHE / f"{cache_tag}.parquet"

    # Check complete cache first
    if final_cache.exists():
        print(f"[Full CE] Loading cached scores from {final_cache}...", flush=True)
        cached = pd.read_parquet(final_cache)
        if len(cached) == len(candidates_df):
            return cached["p_ce"].to_numpy(dtype=np.float32)
        print(f"[Full CE] Cache length mismatch ({len(cached)} vs {len(candidates_df)}), re-scoring...", flush=True)

    t_start = time.time()
    total_pairs = len(candidates_df)
    n_chunks = (total_pairs + chunk_size - 1) // chunk_size

    # Auto-detect finetuned weights
    if model_name == DEFAULT_MODEL:
        finetuned_dir = W / "cross_encoder_finetuned"
        if finetuned_dir.exists():
            model_name = str(finetuned_dir)

    print(f"\n{'='*65}")
    print(f" Option 3: Full Cross-Encoder on ALL {total_pairs:,} Candidate Pairs")
    print(f" Model     : {model_name}")
    print(f" Batch Size: {batch_size} | Chunks: {n_chunks} x {chunk_size:,}")
    print(f"{'='*65}\n")

    # Build text lookup: business_name + " | " + business_address + " [" + country + "]"
    def _build_text_map(df):
        ids = df["entity_id"].to_numpy(dtype=object)
        names = df["business_name"].to_numpy(dtype=object)
        addrs = df.get("business_address", df.get("addr", pd.Series([""] * len(df)))).to_numpy(dtype=object)
        ctry = df["country"].to_numpy(dtype=object)
        texts = np.array([
            f"{n} | {a} [{c}]" for n, a, c in zip(names, addrs, ctry)
        ], dtype=object)
        return dict(zip(ids, texts))

    s1_map = _build_text_map(s1_df)
    pool_map = _build_text_map(pool_df)

    ce = FastCrossEncoder(model_name=model_name)
    all_probs = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_pairs)
        chunk_cache = CACHE / f"{cache_tag}_chunk{chunk_idx}.parquet"

        # Per-chunk resume
        if chunk_cache.exists():
            chunk_p = pd.read_parquet(chunk_cache)["p_ce"].to_numpy(dtype=np.float32)
            if len(chunk_p) == end - start:
                print(f"  Chunk {chunk_idx+1}/{n_chunks} [{start:,}:{end:,}]: cached ({len(chunk_p):,} pairs)", flush=True)
                all_probs.append(chunk_p)
                continue

        print(f"\n  Chunk {chunk_idx+1}/{n_chunks} [{start:,}:{end:,}] ({end-start:,} pairs)...", flush=True)
        chunk_c = candidates_df.iloc[start:end]

        texts_a = [s1_map.get(s, "") for s in chunk_c["s1"].to_numpy(dtype=object)]
        texts_b = [pool_map.get(c, "") for c in chunk_c["cand"].to_numpy(dtype=object)]

        chunk_p = ce.predict_probs(texts_a, texts_b, batch_size=batch_size)
        pd.DataFrame({"p_ce": chunk_p}).to_parquet(chunk_cache, index=False)

        elapsed = time.time() - t_start
        done_pairs = end
        rate = done_pairs / elapsed
        eta_min = (total_pairs - done_pairs) / rate / 60 if rate > 0 else 0
        print(f"  Chunk {chunk_idx+1}/{n_chunks} done | {rate:.0f} pairs/sec | ETA: {eta_min:.1f} min remaining", flush=True)
        all_probs.append(chunk_p)

    final_probs = np.concatenate(all_probs)
    pd.DataFrame({"p_ce": final_probs}).to_parquet(final_cache, index=False)

    # Clean up chunk caches
    for chunk_idx in range(n_chunks):
        chunk_cache = CACHE / f"{cache_tag}_chunk{chunk_idx}.parquet"
        if chunk_cache.exists():
            chunk_cache.unlink()

    total_min = (time.time() - t_start) / 60
    print(f"\n[DONE] Full Cross-Encoder scored {total_pairs:,} pairs in {total_min:.1f} min | Saved to {final_cache}")
    return final_probs

