import sys, pandas as pd
from multiprocessing import Pool
from config import ROOT, W, DATASET
from er_normalize import norm_name, norm_addr

def _row(args):
    n, a, c = args
    full, core, nosp = norm_name(n)
    addr, nums, zips = norm_addr(a, c)
    return full, core, nosp, addr, " ".join(sorted(nums)), " ".join(sorted(zips))

from pathlib import Path

def get_tsv_path(split, i):
    p = DATASET / split / f"{split}_source{i}.tsv"
    if p.exists(): return p
    p_flat = DATASET / f"{split}_source{i}.tsv"
    if p_flat.exists(): return p_flat
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for f in kaggle_input.rglob(f"{split}_source{i}.tsv"):
            if f.is_file(): return f
    return p

import os, gc

def prep(split, i):
    tsv_path = get_tsv_path(split, i)
    if not tsv_path.exists():
        print(f"Skipping {split} s{i} (file not found: {split}_source{i}.tsv)", flush=True)
        return
    out_parquet = W / f"{split}_s{i}.parquet"
    if out_parquet.exists():
        print(f"{split} s{i} already prepared ({out_parquet.name}).", flush=True)
        return
    print(f"Prepping {split} s{i} from {tsv_path}...", flush=True)
    df = pd.read_csv(tsv_path, sep="\t",
                     dtype=str, keep_default_na=False, engine="pyarrow")
    df["country"] = df.country.str.strip().str.lower().replace("", "unk")
    n_workers = int(os.environ.get("ER_PREP_WORKERS", "2"))
    if n_workers > 1:
        with Pool(n_workers) as p:
            res = p.map(_row, zip(df.business_name, df.business_address, df.country), chunksize=20_000)
    else:
        res = [_row(x) for x in zip(df.business_name, df.business_address, df.country)]
    df[["name_full", "core", "nosp", "addr", "nums", "zips"]] = pd.DataFrame(res, index=df.index)
    del res
    df.to_parquet(out_parquet, index=False)
    print(f"done {split} s{i}: {len(df):,} rows", flush=True)
    del df
    gc.collect()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        splits = [sys.argv[1]]
    else:
        splits = ["train"]
        if get_tsv_path("test", 1).exists():
            splits.append("test")
    for split in splits:
        for i in (1, 2, 3):
            prep(split, i)
