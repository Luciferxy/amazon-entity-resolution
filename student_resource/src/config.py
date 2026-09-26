import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"; W.mkdir(exist_ok=True)
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
CACHE = Path(os.environ.get("ER_CACHE", str(W))); CACHE.mkdir(parents=True, exist_ok=True)

def _find_dataset():
    if os.environ.get("ER_DATASET_DIR"):
        p = Path(os.environ["ER_DATASET_DIR"])
        if p.exists(): return p
    if (ROOT / "dataset/train/train_source1.tsv").exists():
        return ROOT / "dataset"
    # Auto-detect Kaggle input directory
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train_source1.tsv"):
            if p.parent.name == "train":
                return p.parent.parent
    return ROOT / "dataset"

DATASET = _find_dataset()

K = int(os.environ.get("ER_K", 5))                  # top-k S1 per pool record per channel
PRUNE = dict(margin=1.0, floor=0.0, max_rank=32)    # retain the union from four retrieval channels
BLOCK_COLS = ["entity_id", "country", "nosp", "addr"]
FEAT_COLS = ["business_name", "name_full", "core", "nums", "zips"]
ALL_COLS = BLOCK_COLS + FEAT_COLS

def load(split, i, cols):
    return pd.read_parquet(W / f"{split}_s{i}.parquet", columns=cols, dtype_backend="pyarrow")

def load_pool(split, cols):
    return pd.concat([load(split, i, cols) for i in (2, 3)], ignore_index=True)

def read_gt():
    return pd.read_csv(DATASET / "train/train_ground_truth.tsv", sep="\t",
                       dtype=str, keep_default_na=False)

def validation_s1_ids(gt, n=50_000):
    """Deterministic Source-1 holdout shared by the Kaggle recall and train steps."""
    return set(gt.source1_entity_id.sample(min(n, len(gt)), random_state=0).tolist())

def gt_pairs(gt):
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex.loc[ex.m != "", ["source1_entity_id", "m"]]
    return ex.rename(columns={"source1_entity_id": "s1", "m": "cand"}).reset_index(drop=True)
