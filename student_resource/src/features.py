import os, multiprocessing as mp, numpy as np, pandas as pd
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"])
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

TXT = ["business_name", "name_full", "core", "nosp", "addr", "nums", "zips"]
_G = {}

def _jac(a, b):
    a, b = set(a.split()), set(b.split()); u = a | b
    return len(a & b) / len(u) if u else -1.0

def _first(a, b):
    a, b = a.split(), b.split()
    return float(bool(a) and bool(b) and a[0] == b[0])

def _lenr(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0

def _chunk(bounds):
    lo, hi = bounds
    s1, pool = _G["s1"], _G["pool"]
    ls, rs = _G["lpos"][lo:hi], _G["rpos"][lo:hi]
    L = {k: s1[k].iloc[ls].tolist() for k in TXT}
    R = {k: pool[k].iloc[rs].tolist() for k in TXT}
    n = hi - lo
    f = lambda fn, col: np.fromiter((fn(a, b) for a, b in zip(L[col], R[col])), np.float32, n)
    F = {"n_tsr": f(fuzz.token_set_ratio, "core"), "n_tsort": f(fuzz.token_sort_ratio, "core"),
         "n_ratio": f(fuzz.ratio, "name_full"), "n_part": f(fuzz.partial_ratio, "nosp"),
         "n_jw": f(JaroWinkler.similarity, "nosp"), "n_jac": f(_jac, "core"),
         "n_first": f(_first, "core"), "n_lenr": f(_lenr, "nosp"),
         "a_tsr": f(fuzz.token_set_ratio, "addr"), "a_part": f(fuzz.partial_ratio, "addr"),
         "a_jac": f(_jac, "addr"), "num_jac": f(_jac, "nums"), "zip_jac": f(_jac, "zips")}
    has = lambda side, col: np.fromiter((x != "" for x in side[col]), bool, n)
    ln, rn, lz, rz = has(L, "nums"), has(R, "nums"), has(L, "zips"), has(R, "zips")
    F["num_conflict"] = (ln & rn & (F["num_jac"] == 0)).astype(np.int8)
    F["zip_conflict"] = (lz & rz & (F["zip_jac"] == 0)).astype(np.int8)
    F["l_addr_missing"] = (~has(L, "addr")).astype(np.int8)
    F["r_addr_missing"] = (~has(R, "addr")).astype(np.int8)
    F["r_nums_missing"] = (~rn).astype(np.int8)
    F["r_nonascii"] = np.fromiter((not x.isascii() for x in R["business_name"]), bool, n).astype(np.int8)
    F["name_only_match"] = ((F["r_addr_missing"] == 1) & (F["n_jw"] >= 0.85)).astype(np.int8)
    F["addr_only_match"] = ((F["r_nonascii"] == 1) & (F["a_tsr"] >= 80)).astype(np.int8)
    F["both_high"] = ((F["n_tsr"] >= 80) & (F["a_tsr"] >= 80)).astype(np.int8)
    F["exact_num_match"] = ((F["num_jac"] == 1.0) & (F["num_conflict"] == 0)).astype(np.int8)
    F["exact_zip_match"] = ((F["zip_jac"] == 1.0) & (F["zip_conflict"] == 0)).astype(np.int8)
    return pd.DataFrame(F)

def features(c, s1, pool, n_jobs=None, step=200_000):
    c = c.reset_index(drop=True)
    lpos = pd.Index(s1["entity_id"].to_numpy(dtype=object)).get_indexer(c["s1"].to_numpy(dtype=object))
    rpos = pd.Index(pool["entity_id"].to_numpy(dtype=object)).get_indexer(c["cand"].to_numpy(dtype=object))
    assert (lpos >= 0).all() and (rpos >= 0).all(), "ids missing from s1/pool"
    _G.update(s1=s1, pool=pool, lpos=lpos, rpos=rpos)
    bounds = [(i, min(i + step, len(c))) for i in range(0, len(c), step)]
    n = n_jobs if n_jobs is not None else int(os.environ.get("ER_FEAT_JOBS", "1"))
    if n > 1 and len(bounds) > 1:
        with mp.get_context("fork").Pool(n) as p:
            parts = p.map(_chunk, bounds)
    else:
        parts = [_chunk(b) for b in bounds]
    F = pd.concat(parts, ignore_index=True)
    for col in ("sim_name", "sim_addr", "sim_name_word", "sim_addr_word",
                "score", "best", "rank", "gap"):
        F[col] = c[col].to_numpy() if col in c.columns else np.zeros(len(c), dtype=np.float32)
    if "n_cands" in c.columns:
        F["n_cands"] = c["n_cands"].to_numpy()
        F["score_vs_best_s1"] = c["score_vs_best_s1"].to_numpy()
        F["is_s3"] = c["is_s3"].to_numpy()
    else:
        g = c.groupby("s1")["score"]
        F["n_cands"] = g.transform("size").to_numpy()
        F["score_vs_best_s1"] = (c["score"] - g.transform("max")).to_numpy()
        F["is_s3"] = c["cand"].str.startswith("S3-").to_numpy().astype(np.int8)
    _G.clear()
    return F
