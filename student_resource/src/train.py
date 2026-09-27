try:
    import sparse_dot_topn
except ImportError:
    pass
import os, sys, gc, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from config import W, K, PRUNE, ALL_COLS, load, load_pool, read_gt, gt_pairs
from block import block, prune
from features import features
from decide import decide
from metrics import macro_f05

PARAMS = dict(
    objective="binary",
    learning_rate=0.06,
    num_leaves=127,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.5,
    verbose=-1,
    num_threads=os.cpu_count(),
)

def main(n_s1):
    t0 = time.time()
    print("=" * 65)
    print(f" 🚀 Training Entity Resolution LightGBM Model on {n_s1:,} S1 Entities")
    print("=" * 65)

    gt = read_gt()
    pairs = gt_pairs(gt)
    sample = gt.source1_entity_id.sample(min(n_s1, len(gt)), random_state=1).tolist()
    sample_set = set(sample)
    sp = pairs[pairs.s1.isin(sample_set)]

    # Filter S1 to sample entities to keep index tight, fast, and 100% aligned
    s1_all = load("train", 1, ALL_COLS)
    s1 = s1_all[s1_all.entity_id.isin(sample_set)].reset_index(drop=True)
    del s1_all
    gc.collect()

    pool = load_pool("train", ALL_COLS)
    is_matched = pool.entity_id.isin(pairs.cand)
    is_selected_true = pool.entity_id.isin(sp.cand)
    q_true = pool[is_selected_true]

    # Balanced hard negative sampling (1.5x negatives with 70% hard negatives from other companies)
    n_neg = int(1.5 * len(q_true))
    n_hard = int(0.70 * n_neg)
    q_hard = pool[is_matched & ~is_selected_true].sample(min(n_hard, int((is_matched & ~is_selected_true).sum())), random_state=1)
    q_unmatched = pool[~is_matched].sample(min(n_neg - len(q_hard), int((~is_matched).sum())), random_state=1)
    q = pd.concat([q_true, q_hard, q_unmatched], ignore_index=True)
    del pool, q_true, q_hard, q_unmatched, is_matched, is_selected_true
    gc.collect()
    print(f"[1/5] Loaded in {time.time()-t0:.1f}s | S1: {len(s1):,} | Candidate pool: {len(q):,}", flush=True)

    # 2. Blocking & Pruning
    t_b = time.time()
    cache_name = f"train{n_s1}_balanced"
    c = prune(block(s1, q, K, cache=cache_name), **PRUNE)
    y = c.merge(pairs.assign(y=1), on=["s1", "cand"], how="left")["y"].fillna(0).to_numpy(np.int8)
    print(f"[2/5] Blocking in {time.time()-t_b:.1f}s | Pairs: {len(c):,} | Positives: {y.sum():,} ({y.mean():.1%}) | Recall: {y.sum()/len(sp):.4f}", flush=True)

    # 3. Vectorized Feature Extraction
    t_f = time.time()
    F = features(c, s1, q)
    print(f"[3/5] Extracted {len(F.columns)} features in {time.time()-t_f:.1f}s", flush=True)

    truth = {s: set() for s in sample}
    for s, m in zip(sp.s1.to_numpy(), sp.cand.to_numpy()):
        truth[s].add(m)

    # 4. 4-Fold GroupKFold Cross-Validation
    print("\n[4/5] Running 4-Fold GroupKFold Cross-Validation...", flush=True)
    groups = c.s1.to_numpy()
    oof = np.zeros(len(c), dtype=np.float32)
    iters = []

    for fold, (tr, va) in enumerate(GroupKFold(4).split(F, y, groups)):
        t_fold = time.time()
        m = lgb.train(
            PARAMS,
            lgb.Dataset(F.iloc[tr], y[tr]),
            3000,
            valid_sets=[lgb.Dataset(F.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  Fold {fold+1}/4: best iter = {m.best_iteration} in {time.time()-t_fold:.1f}s", flush=True)

    rounds = int(np.mean(iters) * 1.1)
    print(f"Optimal training rounds: {rounds} (from fold mean {int(np.mean(iters))})", flush=True)

    # 5. Threshold Grid Search for Macro F0.5
    print("\n[5/5] Optimizing decision threshold for Macro F0.5 on OOF predictions...", flush=True)
    def score(cc, p, ids, prm):
        pred = decide(cc, p, ids, **prm)
        return macro_f05({s: set(v) for s, v in pred.items()}, {s: truth[s] for s in ids})

    grid = (
        [dict(mode="thr", thr=float(t)) for t in np.arange(0.35, 0.85, 0.05)] +
        [dict(mode="ef", pmin=float(pm), ceil=float(ce)) for pm in (0.25, 0.35, 0.45, 0.55) for ce in (0.95, 0.99)]
    )
    res = sorted(((score(c, oof, sample, g), g) for g in grid), key=lambda x: -x[0])
    for sc, g in res[:6]:
        print(f"  OOF macro F0.5 = {sc:.4f}  {g}", flush=True)
    best = res[0][1]
    print(f"\n🏆 Best Decision Strategy: {best} (OOF F0.5 = {res[0][0]:.4f})", flush=True)

    # Leave-One-Country-Out Evaluation
    print("\n--- Leave-One-Country-Out Cross-Check ---", flush=True)
    cmap = dict(zip(s1.entity_id.to_numpy(dtype=object), s1.country.to_numpy(dtype=object)))
    ctry = c.s1.map(cmap).to_numpy()
    for A in pd.unique(ctry):
        tr, te = ctry != A, ctry == A
        if tr.sum() == 0: continue
        m_loco = lgb.train(PARAMS, lgb.Dataset(F[tr], y[tr]), rounds)
        ids = [s for s in sample if cmap[s] == A]
        loco_sc = score(c[te], m_loco.predict(F[te]), ids, best)
        print(f"  LOCO Train on others -> Test {A}: Macro F0.5 = {loco_sc:.4f}", flush=True)

    # Train Final Model on All Data & Save
    print(f"\nTraining final model on all {len(F):,} pairs ({rounds} rounds)...", flush=True)
    final = lgb.train(PARAMS, lgb.Dataset(F, y), rounds)
    final.save_model(str(W / "lgb.txt"))
    print(f"Saved model to {W / 'lgb.txt'}", flush=True)

    model_metadata = dict(
        features=list(F.columns),
        decision=best,
        prune=PRUNE,
        K=K,
        rounds=rounds,
        n_s1=n_s1,
        oof_f05=float(res[0][0]),
    )
    json.dump(model_metadata, open(W / "model_meta.json", "w"), indent=2, default=float)
    print(f"Saved metadata to {W / 'model_meta.json'}", flush=True)

    imp = pd.Series(final.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print("\nTop 15 Most Important Features:\n" + imp.head(15).to_string())
    print("\n" + "="*65)
    print(f"🎉 Training Complete in {time.time()-t0:.1f}s ({((time.time()-t0)/60):.1f} min)!")
    print("="*65 + "\n")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    main(n)
