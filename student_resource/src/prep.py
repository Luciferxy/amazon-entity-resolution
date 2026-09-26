import sys, pandas as pd
from multiprocessing import Pool
from config import ROOT, W
from er_normalize import norm_name, norm_addr

def _row(args):
    n, a, c = args
    full, core, nosp = norm_name(n)
    addr, nums, zips = norm_addr(a, c)
    return full, core, nosp, addr, " ".join(sorted(nums)), " ".join(sorted(zips))

def prep(split, i):
    df = pd.read_csv(ROOT / "dataset" / split / f"{split}_source{i}.tsv", sep="\t",
                     dtype=str, keep_default_na=False, engine="pyarrow")
    df["country"] = df.country.str.strip().str.lower().replace("", "unk")
    with Pool() as p:
        res = p.map(_row, zip(df.business_name, df.business_address, df.country), chunksize=20_000)
    df[["name_full", "core", "nosp", "addr", "nums", "zips"]] = pd.DataFrame(res, index=df.index)
    df.to_parquet(W / f"{split}_s{i}.parquet", index=False)
    print(f"done {split} s{i}: {len(df):,} rows", flush=True)

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    for i in (1, 2, 3):
        prep(split, i)
