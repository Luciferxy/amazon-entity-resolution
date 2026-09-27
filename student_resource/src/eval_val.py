"""
Validation Scorer for Business Entity Resolution
================================================
Computes the exact Macro F0.5 score, Precision, and Recall on the
held-out validation split of the training dataset.
"""

import sys, os, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

from config import ROOT, W, ALL_COLS, read_gt, gt_pairs, validation_s1_ids, load, load_pool
from metrics import macro_f05
from features import features
from decide import decide
from block import block_country, prune

def main(n_val=5000):
    print("=" * 65)
    print(" 📊 Entity Resolution Validation Scorer (Macro F0.5)")
    print("=" * 65)

    if not (W / "train_s1.parquet").exists():
        print("\n[INFO] train_s1.parquet not found in work/. Prepping 5,000 validation samples...")
        from prep import prep
        prep("train", 1)
        prep("train", 2)
        prep("train", 3)

    gt = read_gt()
    pairs = gt_pairs(gt)
    truth_map = {}
    for s, m in zip(pairs.s1.to_numpy(), pairs.cand.to_numpy()):
        truth_map.setdefault(s, set()).add(m)

    val_s1_all = list(validation_s1_ids(gt))
    val_s1 = val_s1_all[:n_val]
    val_set = set(val_s1)
    val_truth = {s: truth_map.get(s, set()) for s in val_s1}

    print(f"Validation entities: {len(val_s1):,} (Positives: {sum(len(v) > 0 for v in val_truth.values()):,} | Singletons: {sum(len(v) == 0 for v in val_truth.values()):,})")

    meta = json.load(open(W / "model_meta.json"))
    model = lgb.Booster(model_file=str(W / "lgb.txt"))
    feature_names = meta["features"]
    decision_params = meta.get("decision", {"mode": "ef", "pmin": 0.15, "ceil": 0.99})

    s1 = load("train", 1, ALL_COLS)
    s1_val = s1[s1.entity_id.isin(val_set)].reset_index(drop=True)
    del s1

    # Load candidates from validation cache or evaluate
    print("\nEvaluating model predictions on validation split...")
    # S1 countries
    countries = list(pd.unique(s1_val["country"]))
    all_preds = {}

    for ctry in countries:
        s1_c = s1_val[s1_val.country == ctry]
        pool_c = load_pool("train", ALL_COLS, country=ctry)
        c_c = block_country(s1_c, pool_c, ctry, k=meta.get("K", 7))
        c_c = prune(c_c, **meta.get("prune", dict(margin=0.7, floor=0.0, max_rank=8)))
        
        g = c_c.groupby("s1")["score"]
        c_c["n_cands"] = g.transform("size").to_numpy(dtype=np.int32)
        c_c["score_vs_best_s1"] = (c_c["score"] - g.transform("max")).to_numpy(dtype=np.float32)
        c_c["is_s3"] = c_c["cand"].str.startswith("S3-").to_numpy(dtype=np.int8)

        F_c = features(c_c, s1_c, pool_c)
        p_c = model.predict(F_c[feature_names])

        ce_mode = os.environ.get("ER_CE_MODE", "none")
        if ce_mode == "cascade":
            from hybrid_cross_encoder import cascade_rescore
            p_c = cascade_rescore(c_c, p_c, s1_c, pool_c, low_thr=0.40, high_thr=0.85, ce_weight=0.65)

        preds = decide(c_c, p_c, s1_c["entity_id"].tolist(), **decision_params)
        all_preds.update(preds)

    # Compute Macro F0.5
    f05 = macro_f05({s: set(v) for s, v in all_preds.items()}, val_truth)

    # Precision & Recall metrics
    tps, fps, fns = 0, 0, 0
    singletons_correct, singletons_total = 0, 0
    for s, truth in val_truth.items():
        pred = set(all_preds.get(s, []))
        tp = len(pred & truth)
        fp = len(pred - truth)
        fn = len(truth - pred)
        tps += tp; fps += fp; fns += fn
        if len(truth) == 0:
            singletons_total += 1
            if len(pred) == 0: singletons_correct += 1

    prec = tps / (tps + fps + 1e-9)
    rec = tps / (tps + fns + 1e-9)

    print("\n" + "=" * 65)
    print(" 🏆 VALIDATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Macro F0.5 Score   :  {f05:.4f}")
    print(f"  Precision (Global) :  {prec:.4f}")
    print(f"  Recall (Global)    :  {rec:.4f}")
    print(f"  Singleton Accuracy :  {singletons_correct / (singletons_total + 1e-9):.2%} ({singletons_correct:,}/{singletons_total:,})")
    print("=" * 65)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(n)
