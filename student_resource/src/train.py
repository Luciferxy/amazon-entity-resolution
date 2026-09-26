try:
    import sparse_dot_topn
except ImportError:
    pass
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from config import W, K, PRUNE, ALL_COLS, load, load_pool, read_gt, gt_pairs
from block import block, prune
from features import features
from decide import decide
from metrics import macro_f05

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=os.cpu_count())

def main(n_s1):
    t0 = time.time()
    gt = read_gt(); pairs = gt_pairs(gt)
    s1 = load("train", 1, ALL_COLS)
    sample = gt.source1_entity_id.sample(n_s1, random_state=1).tolist()
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
    print(f"loaded {time.time()-t0:.0f}s | queries {len(q):,}", flush=True)

    c = prune(block(s1, q, K, cache=f"train{n_s1}_raw"), **PRUNE)
    y = c.merge(pairs.assign(y=1), on=["s1", "cand"], how="left")["y"].fillna(0).to_numpy(np.int8)
    print(f"candidates {len(c):,} | positives {y.sum():,} | recall {y.sum()/len(sp):.4f}", flush=True)

    t0 = time.time(); F = features(c, s1, q); print(f"features {time.time()-t0:.0f}s", flush=True)
    truth = {s: set() for s in sample}
    for s, m in zip(sp.s1.to_numpy(), sp.cand.to_numpy()): truth[s].add(m)

    groups = c.s1.to_numpy(); oof = np.zeros(len(c)); iters = []
    for tr, va in GroupKFold(4).split(F, y, groups):
        m = lgb.train(PARAMS, lgb.Dataset(F.iloc[tr], y[tr]), 3000,
                      valid_sets=[lgb.Dataset(F.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration); iters.append(m.best_iteration)
    rounds = int(np.mean(iters) * 1.1)

    def score(cc, p, ids, prm):
        pred = decide(cc, p, ids, **prm)
        return macro_f05({s: set(v) for s, v in pred.items()}, {s: truth[s] for s in ids})

    grid = ([dict(mode="ef", pmin=pm, ceil=ce) for pm in (0.02, 0.05, 0.1, 0.2, 0.3)
             for ce in (0.9, 0.95, 0.99, 1.0)] + [dict(mode="thr", thr=t) for t in np.arange(0.3, 0.95, 0.05)])
    res = sorted(((score(c, oof, sample, g), g) for g in grid), key=lambda x: -x[0])
    for sc, g in res[:8]: print(f"OOF macro F0.5 {sc:.4f}  {g}")
    best = res[0][1]

    cmap = dict(zip(s1.entity_id.to_numpy(dtype=object), s1.country.to_numpy(dtype=object)))
    ctry = c.s1.map(cmap).to_numpy()
    for A in pd.unique(ctry):                              # leave-one-country-out (France proxy)
        tr, te = ctry != A, ctry == A
        if tr.sum() == 0: continue
        m = lgb.train(PARAMS, lgb.Dataset(F[tr], y[tr]), rounds)
        ids = [s for s in sample if cmap[s] == A]
        print(f"LOCO train on others -> test {A}: macro F0.5 "
              f"{score(c[te], m.predict(F[te]), ids, best):.4f}", flush=True)

    final = lgb.train(PARAMS, lgb.Dataset(F, y), rounds)
    final.save_model(str(W / "lgb.txt"))
    json.dump(dict(features=list(F.columns), decision=best, prune=PRUNE, K=K, rounds=rounds),
              open(W / "model_meta.json", "w"), indent=2, default=float)
    imp = pd.Series(final.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print("\nTop features (gain):\n" + imp.head(20).to_string())

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150_000)
