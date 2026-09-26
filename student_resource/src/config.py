import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"; W.mkdir(exist_ok=True)
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
CACHE = Path(os.environ.get("ER_CACHE", str(W))); CACHE.mkdir(parents=True, exist_ok=True)

K = int(os.environ.get("ER_K", 3))                  # top-k S1 per pool record per channel
PRUNE = dict(margin=1.0, floor=0.0, max_rank=10)    # keep all retrieved pairs; LightGBM ranks them
BLOCK_COLS = ["entity_id", "country", "nosp", "addr"]
FEAT_COLS = ["business_name", "name_full", "core", "nums", "zips"]
ALL_COLS = BLOCK_COLS + FEAT_COLS

def load(split, i, cols):
    return pd.read_parquet(W / f"{split}_s{i}.parquet", columns=cols, dtype_backend="pyarrow")

def load_pool(split, cols):
    return pd.concat([load(split, i, cols) for i in (2, 3)], ignore_index=True)

def read_gt():
    return pd.read_csv(ROOT / "dataset/train/train_ground_truth.tsv", sep="\t",
                       dtype=str, keep_default_na=False)

def gt_pairs(gt):
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex.loc[ex.m != "", ["source1_entity_id", "m"]]
    return ex.rename(columns={"source1_entity_id": "s1", "m": "cand"}).reset_index(drop=True)
