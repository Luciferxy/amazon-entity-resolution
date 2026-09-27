"""
Train Cross-Encoder for Business Entity Resolution (Leaderboard Target: >= 0.990)
================================================================================
Fine-tunes a deep transformer cross-encoder on domain-specific hard positives
and mined hard negatives to eliminate false positives and borderline ambiguities.

Uses:
  - Backbone: cross-encoder/ms-marco-MiniLM-L-6-v2 (or user-specified)
  - Hard negative mining from candidate blocking stage
  - Multi-GPU support (DataParallel on 2x T4 GPUs)
  - Mixed precision (FP16) training
"""

import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import GroupShuffleSplit

from config import ROOT, W, CACHE, DATASET, K, ALL_COLS, load, load_pool, read_gt, gt_pairs
from block import block, prune

MODEL_NAME = os.environ.get("CE_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OUT_DIR = W / "cross_encoder_finetuned"
BATCH_SIZE = int(os.environ.get("CE_BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("CE_EPOCHS", 2))
LR = float(os.environ.get("CE_LR", 2e-5))
MAX_LEN = 96


class EntityPairDataset(Dataset):
    def __init__(self, texts_a, texts_b, labels, tokenizer, max_length=MAX_LEN):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = np.array(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        return self.texts_a[idx], self.texts_b[idx], self.labels[idx]


def collate_train_pairs(batch, tokenizer, max_length=MAX_LEN):
    texts_a, texts_b, labels = zip(*batch)
    enc = tokenizer(
        list(texts_a),
        list(texts_b),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.float32)
    return enc


def mine_pairs(n_sample=80_000):
    """Mine high-quality hard positives and hard negatives for training."""
    print("Mining hard training pairs for Cross-Encoder...", flush=True)
    gt = read_gt()
    pairs = gt_pairs(gt)
    
    # Text columns - must include all blocking and feature columns
    cols = list(dict.fromkeys(ALL_COLS + ["business_address"]))
    s1 = load("train", 1, cols)
    pool = load_pool("train", cols)

    # Sample S1 entities
    sample_s1 = set(gt.source1_entity_id.sample(min(n_sample, len(gt)), random_state=42))
    sp = pairs[pairs.s1.isin(sample_s1)]
    
    # True positives
    tp_df = sp.copy()
    tp_df["label"] = 1.0

    # Look for cached candidate parquet from train
    cached_files = [f for f in CACHE.glob("block_v2_*train*.parquet") if f.is_file()]
    c = None
    if cached_files:
        try:
            print(f"Loading candidate pairs from cache: {[f.name for f in cached_files]}", flush=True)
            candidate_df = pd.concat([pd.read_parquet(f) for f in cached_files], ignore_index=True)
            if "s1" in candidate_df.columns and "cand" in candidate_df.columns:
                c = candidate_df[candidate_df.s1.isin(sample_s1)].reset_index(drop=True)
        except Exception as e:
            print(f"Could not load cache ({e}), falling back to blocking...", flush=True)
            c = None

    if c is None:
        print("Generating candidate pairs via blocking...", flush=True)
        s1_sub = s1[s1.entity_id.isin(sample_s1)].reset_index(drop=True)
        # Filter pool for fast blocking
        is_sp = pool.entity_id.isin(sp.cand)
        q_samp = pd.concat([pool[is_sp], pool[~is_sp].sample(min(100_000, (~is_sp).sum()), random_state=42)])
        c = prune(block(s1_sub, q_samp, k=K), margin=1.0, floor=0.0)

    # Label candidates
    c = c.merge(pairs.assign(is_true=1), on=["s1", "cand"], how="left")
    c["is_true"] = c["is_true"].fillna(0).astype(int)

    # Filter to S1 in sample
    c = c[c.s1.isin(sample_s1)]

    # Hard negatives: non-matching pairs with top similarity scores
    c["score"] = c["score"].astype(np.float32)
    neg_df = c[c.is_true == 0].sort_values("score", ascending=False).groupby("s1").head(2)
    neg_df = neg_df[["s1", "cand"]].copy()
    neg_df["label"] = 0.0

    all_pairs = pd.concat([tp_df[["s1", "cand", "label"]], neg_df[["s1", "cand", "label"]]], ignore_index=True)
    all_pairs = all_pairs.drop_duplicates(subset=["s1", "cand"]).reset_index(drop=True)

    print(f"Total mined pairs: {len(all_pairs):,} (Positives: {(all_pairs.label == 1).sum():,}, Negatives: {(all_pairs.label == 0).sum():,})", flush=True)

    # Build text lookup
    s1_map = dict(zip(
        s1["entity_id"].to_numpy(dtype=object),
        (s1["business_name"] + " | " + s1["business_address"] + " | " + s1["country"]).to_numpy(dtype=object)
    ))
    pool_map = dict(zip(
        pool["entity_id"].to_numpy(dtype=object),
        (pool["business_name"] + " | " + pool["business_address"] + " | " + pool["country"]).to_numpy(dtype=object)
    ))

    # Keep only valid pairs
    valid = all_pairs["s1"].isin(s1_map) & all_pairs["cand"].isin(pool_map)
    all_pairs = all_pairs[valid].reset_index(drop=True)

    texts_a = [s1_map[s] for s in all_pairs["s1"]]
    texts_b = [pool_map[c] for c in all_pairs["cand"]]
    labels = all_pairs["label"].to_numpy(dtype=np.float32)
    groups = all_pairs["s1"].to_numpy(dtype=object)

    return texts_a, texts_b, labels, groups


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"\n=======================================================")
    print(f" Training Cross-Encoder for Entity Resolution")
    print(f" Model Backbone : {MODEL_NAME}")
    print(f" Device         : {device} ({n_gpus} GPUs)")
    print(f" Batch Size     : {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"=======================================================\n")

    texts_a, texts_b, labels, groups = mine_pairs()

    # Train/Validation Group Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(texts_a, labels, groups))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(device)

    if n_gpus > 1 and device == "cuda":
        model = nn.DataParallel(model)

    train_ds = EntityPairDataset(
        [texts_a[i] for i in train_idx],
        [texts_b[i] for i in train_idx],
        labels[train_idx],
        tokenizer,
    )
    val_ds = EntityPairDataset(
        [texts_a[i] for i in val_idx],
        [texts_b[i] for i in val_idx],
        labels[val_idx],
        tokenizer,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_train_pairs(b, tokenizer),
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=lambda b: collate_train_pairs(b, tokenizer),
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    best_f05 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch["labels"].numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        pred_binary = (val_preds >= 0.5).astype(int)

        tp = ((pred_binary == 1) & (val_targets == 1)).sum()
        fp = ((pred_binary == 1) & (val_targets == 0)).sum()
        fn = ((pred_binary == 0) & (val_targets == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec + 1e-9)

        print(f"\n--- Epoch {epoch} Validation ---")
        print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F0.5 Score: {f05:.4f} in {time.time()-t0:.1f}s\n")

        if f05 > best_f05 or epoch == EPOCHS:
            best_f05 = f05
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            # Unwrap DataParallel if used
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"  >>> Saved best model checkpoint to {OUT_DIR} (F0.5 = {best_f05:.4f})", flush=True)

    print(f"\n[DONE] Cross-Encoder training finished! Best Validation F0.5: {best_f05:.4f}\n")


if __name__ == "__main__":
    train()
