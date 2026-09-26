import time, numpy as np, pandas as pd
from config import K, PRUNE, BLOCK_COLS, load, load_pool, read_gt, gt_pairs
from block import block, prune, BACKEND

EXTRA = ["business_name", "business_address"]
t0 = time.time()
gt = read_gt(); pairs = gt_pairs(gt)
s1 = load("train", 1, BLOCK_COLS + EXTRA)
pool = load_pool("train", BLOCK_COLS + EXTRA)
print(f"backend={BACKEND} | loaded in {time.time()-t0:.0f}s", flush=True)

dev = gt.source1_entity_id.sample(50_000, random_state=0)
dev_idx = pd.Index(dev.to_numpy())
dev_pairs = pairs[pairs.s1.isin(dev_idx)]
q = pd.concat([pool[pool.entity_id.isin(dev_pairs.cand)],
               pool[~pool.entity_id.isin(pairs.cand)].sample(int(0.35 * len(dev_pairs)), random_state=0)],
              ignore_index=True)
del pool
print(f"queries: {len(q):,} ({len(dev_pairs):,} true + {len(q)-len(dev_pairs):,} distractors)", flush=True)

t0 = time.time(); raw = block(s1, q, K, cache="dev_raw")
print(f"blocking: {time.time()-t0:.0f}s | raw pairs {len(raw):,}", flush=True)

T = dev_pairs.groupby("s1").size().reindex(dev_idx, fill_value=0).to_numpy()

def report(c, tag):
    cd = c[c.s1.isin(dev_idx)]
    tp = cd.merge(dev_pairs, on=["s1", "cand"]).groupby("s1").size().reindex(dev_idx, fill_value=0).to_numpy()
    nc = cd.groupby("s1").size().reindex(dev_idx, fill_value=0).to_numpy()
    R = np.divide(tp, T, out=np.zeros(len(T)), where=T > 0)
    f = np.where(T == 0, 1.0, np.where(tp > 0, 1.25 * R / (0.25 + R), 0.0))
    print(f"{tag:<34} recall={tp.sum()/T.sum():.4f}  avg_cands={nc.mean():.2f}  oracle_F05={f.mean():.4f}")

report(raw, "raw (no pruning)")
for margin in (0.0, 0.05, 0.10, 0.20):
    for floor in (0.0, 0.15, 0.25):
        report(prune(raw, margin, floor), f"margin={margin} floor={floor}")

c = prune(raw, **PRUNE)
m = dev_pairs.merge(c[["s1", "cand"]], on=["s1", "cand"], how="left", indicator=True)
miss = m[m["_merge"] == "left_only"][["s1", "cand"]]
in_raw = miss.merge(raw[["s1", "cand"]], on=["s1", "cand"]).shape[0]
mq = q[q.entity_id.isin(miss.cand)]
print(f"\nmisses: {len(miss):,} | retrieved-but-pruned: {in_raw:,} | never retrieved: {len(miss)-in_raw:,}")
print(mq["country"].value_counts().to_string())
print(f"missing-address share: {(mq['addr'] == '').mean():.2%} | "
      f"S3 share: {miss.cand.str.startswith('S3-').mean():.2%}")
s1d, qd = s1.set_index("entity_id"), q.set_index("entity_id")
for s, mm in miss.head(10).itertuples(index=False):
    print("\nS1  :", s1d.loc[s, "business_name"], "|", s1d.loc[s, "business_address"])
    print("MISS:", qd.loc[mm, "business_name"], "|", qd.loc[mm, "business_address"])
