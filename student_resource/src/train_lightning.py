"""
Lightning AI Training Script for Business Entity Resolution
==========================================================
Trains entity resolution models with GPU acceleration on Lightning AI:
  - GPU-accelerated sparse candidate blocking (top-k cosine)
  - 28-dimensional dense tabular feature engineering
  - Gradient Boosted Trees (LightGBM)
  - Optional Deep Residual Neural Matcher (PyTorch Lightning)
  - F0.5-optimized decision thresholding and LOCO cross-validation

Usage:
  python3 src/train_lightning.py [N_S1] [--use-nn] [--gpu]
"""

try:
    import sparse_dot_topn
except ImportError:
    pass

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from config import W, K, PRUNE, ALL_COLS, load, load_pool, read_gt, gt_pairs
from block import block, prune
from features import features
from decide import decide
from metrics import macro_f05

# PyTorch & Lightning
try:
    import torch
    import lightning as L
    from lightning_matcher import ERLightningModule, ERDataset
    from torch.utils.data import DataLoader
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import torch
        import pytorch_lightning as L
        from lightning_matcher import ERLightningModule, ERDataset
        from torch.utils.data import DataLoader
        LIGHTNING_AVAILABLE = True
    except ImportError:
        LIGHTNING_AVAILABLE = False


PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=os.cpu_count(),
)


def train_lightning_nn(F_train, y_train, F_val, y_val, in_features, epochs=10, batch_size=2048):
    """Train Deep Residual Neural Matcher with PyTorch Lightning."""
    if not LIGHTNING_AVAILABLE:
        print("  [Warning] Lightning not installed. Skipping NN training.")
        return None

    print(f"\n--- Training PyTorch Lightning Neural Matcher ({epochs} epochs) ---")
    device = "gpu" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    train_ds = ERDataset(F_train.to_numpy(dtype=np.float32), y_train)
    val_ds = ERDataset(F_val.to_numpy(dtype=np.float32), y_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(accelerator == "gpu")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(accelerator == "gpu")
    )

    pos_rate = float(y_train.mean())
    pos_weight = min(10.0, max(1.0, (1.0 - pos_rate) / (pos_rate + 1e-5) * 0.25))

    model = ERLightningModule(
        in_features=in_features,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.2,
        lr=1e-3,
        pos_weight=pos_weight,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)
    return model


def main(n_s1=150_000, use_nn=False, force_gpu=True):
    t_start = time.time()
    if force_gpu and "ER_BACKEND" not in os.environ:
        if torch.cuda.is_available():
            os.environ["ER_BACKEND"] = "gpu"
            os.environ["ER_GPU_MODE"] = "exact"
            print(f"[Device] Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[Device] No CUDA device detected; falling back to CPU.")

    print(f"===============================================================")
    print(f" Lightning AI Training Pipeline | S1 entities: {n_s1:,} | K: {K}")
    print(f" Backend: {os.environ.get('ER_BACKEND', 'cpu')} | NN Matcher: {use_nn}")
    print(f"===============================================================")

    # 1. Load Data
    t0 = time.time()
    gt = read_gt()
    pairs = gt_pairs(gt)
    s1 = load("train", 1, ALL_COLS)

    sample = gt.source1_entity_id.sample(min(n_s1, len(gt)), random_state=1).tolist()
    sp = pairs[pairs.s1.isin(sample)]

    pool = load_pool("train", ALL_COLS)
    is_matched = pool.entity_id.isin(pairs.cand)
    is_selected_true = pool.entity_id.isin(sp.cand)
    q_true = pool[is_selected_true]
    n_neg = int(0.35 * len(q_true))
    n_hard = n_neg // 2
    q_hard = pool[is_matched & ~is_selected_true].sample(n_hard, random_state=1)
    q_unmatched = pool[~is_matched].sample(n_neg - n_hard, random_state=1)
    q = pd.concat([q_true, q_hard, q_unmatched], ignore_index=True)
    del pool, q_true, q_hard, q_unmatched, is_matched, is_selected_true
    print(f"[1/5] Loaded data in {time.time()-t0:.1f}s | Sample queries: {len(q):,}", flush=True)

    # 2. Blocking & Pruning
    t0 = time.time()
    cache_tag = f"train{n_s1}_raw"
    c = prune(block(s1, q, K, cache=cache_tag), **PRUNE)
    y = c.merge(pairs.assign(y=1), on=["s1", "cand"], how="left")["y"].fillna(0).to_numpy(np.int8)
    recall = y.sum() / len(sp) if len(sp) > 0 else 0.0
    print(f"[2/5] Blocking in {time.time()-t0:.1f}s | Pairs: {len(c):,} | Hits: {y.sum():,} | Recall: {recall:.4f}", flush=True)

    # 3. Feature Extraction
    t0 = time.time()
    F = features(c, s1, q)
    feature_names = list(F.columns)
    print(f"[3/5] Extracted {len(feature_names)} features in {time.time()-t0:.1f}s", flush=True)

    # 4. Cross-Validation & Model Training
    t0 = time.time()
    truth = {s: set() for s in sample}
    for s, m in zip(sp.s1.to_numpy(), sp.cand.to_numpy()):
        truth[s].add(m)

    groups = c.s1.to_numpy()
    oof_lgb = np.zeros(len(c))
    iters = []

    print("\n--- 4-Fold GroupKFold Cross-Validation ---")
    gkf = GroupKFold(4)
    for fold, (tr, va) in enumerate(gkf.split(F, y, groups)):
        m = lgb.train(
            PARAMS,
            lgb.Dataset(F.iloc[tr], y[tr]),
            3000,
            valid_sets=[lgb.Dataset(F.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  Fold {fold+1}: best iteration = {m.best_iteration}")

    rounds = int(np.mean(iters) * 1.1)
    oof_preds = oof_lgb

    # Optional: PyTorch Lightning Model
    nn_model = None
    if use_nn and LIGHTNING_AVAILABLE:
        # Train Lightning NN on full training fold
        train_idx, val_idx = next(gkf.split(F, y, groups))
        nn_model = train_lightning_nn(
            F.iloc[train_idx], y[train_idx],
            F.iloc[val_idx], y[val_idx],
            in_features=len(feature_names),
            epochs=10,
        )
        if nn_model is not None:
            nn_oof = nn_model.predict_probs(F.iloc[val_idx].to_numpy(dtype=np.float32))
            print(f"  Lightning NN validation set scored.")
            # Blend 70% LightGBM + 30% Lightning NN
            # (used when predicting)

    # 5. Threshold Tuning for Macro F0.5
    def score(cc, p, ids, prm):
        pred = decide(cc, p, ids, **prm)
        return macro_f05({s: set(v) for s, v in pred.items()}, {s: truth[s] for s in ids})

    grid = (
        [dict(mode="ef", pmin=pm, ceil=ce) for pm in (0.02, 0.05, 0.1, 0.2, 0.3) for ce in (0.9, 0.95, 0.99, 1.0)]
        + [dict(mode="thr", thr=t) for t in np.arange(0.20, 0.85, 0.05)]
    )
    res = sorted(((score(c, oof_preds, sample, g), g) for g in grid), key=lambda x: -x[0])

    print("\n--- Decision Rule Optimization (Macro F0.5) ---")
    for sc, g in res[:6]:
        print(f"  OOF Macro F0.5: {sc:.4f}  |  Params: {g}")
    best = res[0][1]

    # LOCO Check (India vs US)
    cmap = dict(zip(s1.entity_id.to_numpy(dtype=object), s1.country.to_numpy(dtype=object)))
    ctry = c.s1.map(cmap).to_numpy()
    print("\n--- Leave-One-Country-Out (LOCO) Robustness Check ---")
    for A in pd.unique(ctry):
        tr, te = ctry != A, ctry == A
        if tr.sum() == 0:
            continue
        m = lgb.train(PARAMS, lgb.Dataset(F[tr], y[tr]), rounds)
        ids = [s for s in sample if cmap[s] == A]
        s_val = score(c[te], m.predict(F[te]), ids, best)
        print(f"  Trained on others -> Test on {A}: Macro F0.5 = {s_val:.4f}")

    # Final Model Training
    print(f"\n--- Retraining Final Model ({rounds} trees) ---")
    final_lgb = lgb.train(PARAMS, lgb.Dataset(F, y), rounds)
    final_lgb.save_model(str(W / "lgb.txt"))

    # Save metadata
    meta = dict(
        features=feature_names,
        decision=best,
        prune=PRUNE,
        K=K,
        rounds=rounds,
        best_f05=float(res[0][0]),
        use_nn=bool(use_nn and nn_model is not None),
    )
    with open(W / "model_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=float)

    # Feature Importance
    imp = pd.Series(final_lgb.feature_importance("gain"), index=feature_names).sort_values(ascending=False)
    print("\nTop 15 Feature Importances (Gain):")
    for fname, val in imp.head(15).items():
        print(f"  {fname:<20} {val:,.1f}")

    total_time = time.time() - t_start
    print(f"\n[DONE] Training complete in {total_time/60:.1f} min! Best OOF F0.5: {res[0][0]:.4f}")
    print(f"Saved: {W / 'lgb.txt'} and {W / 'model_meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightning AI ER Training")
    parser.add_argument("sample", type=int, nargs="?", default=150_000, help="Number of S1 train entities (default: 150000)")
    parser.add_argument("--use-nn", action="store_true", help="Train PyTorch Lightning neural matcher alongside LightGBM")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    args = parser.parse_args()

    if args.cpu:
        os.environ["ER_BACKEND"] = "cpu"
    else:
        if torch.cuda.is_available():
            os.environ["ER_BACKEND"] = "gpu"

    main(n_s1=args.sample, use_nn=args.use_nn)
