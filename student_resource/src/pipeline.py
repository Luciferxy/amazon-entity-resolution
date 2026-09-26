"""
Optimized Entity Resolution Pipeline
=====================================
Single-file pipeline: prep → block → features → train → predict.

Usage:
    python3 src/pipeline.py             # full run: train + predict
    python3 src/pipeline.py train       # train only (saves model)
    python3 src/pipeline.py predict     # predict only (loads saved model)
    python3 src/pipeline.py eval N      # eval blocking recall on N S1 entities
"""
import os, sys, json, re, time, warnings
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
try:
    import sparse_dot_topn
except ImportError:
    pass
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work";  WORK.mkdir(exist_ok=True)
OUT  = ROOT / "output"; OUT.mkdir(exist_ok=True)

# ─── Normalization ──────────────────────────────────────────────────────────
LEGAL = {"inc":"inc","incorporated":"inc","corp":"corp","corporation":"corp",
         "co":"co","company":"co","llc":"llc","llp":"llp","ltd":"ltd",
         "limited":"ltd","pvt":"pvt","private":"pvt","plc":"plc",
         "sarl":"sarl","sas":"sas","sa":"sa","eurl":"eurl","sci":"sci",
         "gmbh":"gmbh","ag":"ag","prop":"prop","proprietorship":"prop"}
LEGAL_SET = set(LEGAL.values())

ADDR_ABBR = {"road":"rd","street":"st","avenue":"ave","av":"ave","drive":"dr",
             "place":"pl","terrace":"ter","boulevard":"blvd","bd":"blvd",
             "lane":"ln","court":"ct","highway":"hwy","township":"twp",
             "apartment":"apt","suite":"ste","building":"bldg","floor":"fl",
             "sector":"sec","near":"nr","opposite":"opp",
             "north":"n","south":"s","east":"e","west":"w","rue":"rue"}

STATE_US = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar",
    "california":"ca","colorado":"co","connecticut":"ct","delaware":"de",
    "district of columbia":"dc","florida":"fl","georgia":"ga","hawaii":"hi",
    "idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks",
    "kentucky":"ky","louisiana":"la","maine":"me","maryland":"md",
    "massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms",
    "missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv",
    "new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny",
    "north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
    "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc",
    "south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt",
    "virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy"}

STATE_IN = {"andhra pradesh":"ap","arunachal pradesh":"ar","assam":"as","bihar":"br",
    "chhattisgarh":"cg","chattisgarh":"cg","goa":"ga","gujarat":"gj","haryana":"hr",
    "himachal pradesh":"hp","jharkhand":"jh","karnataka":"ka","kerala":"kl",
    "madhya pradesh":"mp","maharashtra":"mh","manipur":"mn","meghalaya":"ml","mizoram":"mz",
    "nagaland":"nl","odisha":"od","orissa":"od","punjab":"pb","rajasthan":"rj","sikkim":"sk",
    "tamil nadu":"tn","tamilnadu":"tn","telangana":"ts","tripura":"tr","uttar pradesh":"up",
    "uttarakhand":"uk","uttaranchal":"uk","west bengal":"wb","andaman and nicobar islands":"an",
    "chandigarh":"ch","dadra and nagar haveli and daman and diu":"dd","delhi":"dl",
    "jammu and kashmir":"jk","ladakh":"la","lakshadweep":"ld","puducherry":"py","pondicherry":"py"}

ALL_STATES = {**STATE_US, **STATE_IN}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ALL_STATES), key=len, reverse=True)) + r")\b")
NULLS = {"null", "none", "nan", "na", "nil"}
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|biz|in|co\.in|fr)$")

def _base(s):
    if not isinstance(s, str) or not s: return ""
    s = anyascii(s).lower().strip().replace("&", " and ")
    s = DOMAIN_RE.sub(r"\\1", s)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\\1", s)
    return re.sub(r"[^a-z0-9 ]+", " ", s)

def norm_name(s):
    t = [LEGAL.get(w, w) for w in _base(s).split()]
    core = [w for w in t if w not in LEGAL_SET] or t
    return " ".join(t), " ".join(core), "".join(core)

def norm_addr(s, country):
    s = _base(s)
    s = _STATE_RE.sub(lambda m: ALL_STATES[m.group(1)], s)
    t = [ADDR_ABBR.get(w, w) for w in s.split() if w not in NULLS]
    nums = {w for w in t if w.isdigit()}
    zips = {w for w in nums if len(w) in (5, 6)}
    return " ".join(sorted(t)), " ".join(sorted(nums)), " ".join(sorted(zips))

def _prep_row(args):
    n, a, c = args
    full, core, nosp = norm_name(n)
    addr, nums, zips = norm_addr(a, c)
    return full, core, nosp, addr, nums, zips

# ─── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(split):
    for i in (1, 2, 3):
        out = WORK / f"{split}_s{i}.parquet"
        if out.exists():
            print(f"  {out.name} exists, skipping", flush=True)
            continue
        df = pd.read_csv(ROOT / "dataset" / split / f"{split}_source{i}.tsv",
                         sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
        df["country"] = df.country.str.strip().str.lower().replace("", "unk")
        with Pool() as p:
            res = p.map(_prep_row, zip(df.business_name, df.business_address, df.country),
                        chunksize=20_000)
        df[["name_full","core","nosp","addr","nums","zips"]] = pd.DataFrame(res, index=df.index)
        df.to_parquet(out, index=False)
        print(f"  done {split} s{i}: {len(df):,} rows", flush=True)

def load(split, i, cols=None):
    df = pd.read_parquet(WORK / f"{split}_s{i}.parquet")
    return df[cols] if cols else df

def load_pool(split, cols=None):
    return pd.concat([load(split, i, cols) for i in (2, 3)], ignore_index=True)

def read_gt():
    return pd.read_csv(ROOT / "dataset/train/train_ground_truth.tsv", sep="\t",
                       dtype=str, keep_default_na=False)

def gt_pairs(gt):
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex.loc[ex.m != "", ["source1_entity_id", "m"]]
    return ex.rename(columns={"source1_entity_id": "s1", "m": "cand"}).reset_index(drop=True)

# ─── Blocking (inverted index + TF-IDF) ────────────────────────────────────
def _char_ngrams(s, n=3):
    """Generate character n-grams from string."""
    s = s.strip()
    if len(s) < n: return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def _word_ngrams(s, n=2):
    """Generate word n-grams."""
    words = s.split()
    if len(words) < n: return [s] if s else []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def _block_keys(nosp, core, addr, nums):
    """Generate multiple blocking keys for a record."""
    keys = set()
    # 1) Exact nospace name
    if nosp and len(nosp) >= 4:
        keys.add(("nosp", nosp))
    # 2) Core name word bigrams
    words = core.split()
    if len(words) >= 2:
        for i in range(len(words) - 1):
            bg = words[i] + " " + words[i+1]
            if len(bg) >= 5:
                keys.add(("bg", bg))
    elif len(words) == 1 and len(words[0]) >= 4:
        keys.add(("bg", words[0]))
    # 3) Sorted first 3 chars of each core word (if >=2 words)
    if len(words) >= 2:
        prefixes = tuple(sorted(w[:3] for w in words if len(w) >= 3))
        if len(prefixes) >= 2:
            keys.add(("pfx", prefixes))
    # 4) Char trigrams of nosp (top discriminative ones only)
    if nosp and len(nosp) >= 6:
        grams = _char_ngrams(nosp, 4)
        for g in grams[:8]:  # limit to first 8 4-grams
            keys.add(("c4", g))
    # 5) Address numbers (if any meaningful ones)
    num_list = [n for n in nums.split() if n and len(n) <= 5]
    if num_list:
        for n in num_list[:3]:
            keys.add(("num", n))
    return keys

def block_inverted(s1, pool, max_cands_per_s1=30):
    """
    Inverted-index blocking: build index from S1, probe with pool records.
    Much faster than TF-IDF all-pairs on CPU.
    """
    t0 = time.time()
    # Build inverted index from S1 per country
    s1_by_country = {}
    for c in pd.unique(s1.country.to_numpy(dtype=object)):
        s1c = s1[s1.country == c]
        idx = {}  # key -> list of s1 positions
        for pos, (nosp, core, addr, nums) in enumerate(
                zip(s1c.nosp.to_numpy(dtype=object),
                    s1c.core.to_numpy(dtype=object),
                    s1c.addr.to_numpy(dtype=object),
                    s1c.nums.to_numpy(dtype=object))):
            for k in _block_keys(nosp, core, addr, nums):
                idx.setdefault(k, []).append(pos)
        s1_by_country[c] = (s1c, idx)
    print(f"  index built in {time.time()-t0:.0f}s", flush=True)

    # Probe with pool records
    t0 = time.time()
    results = []
    pool_ids = pool.entity_id.to_numpy(dtype=object)
    pool_nosp = pool.nosp.to_numpy(dtype=object)
    pool_core = pool.core.to_numpy(dtype=object)
    pool_addr = pool.addr.to_numpy(dtype=object)
    pool_nums = pool.nums.to_numpy(dtype=object)
    pool_ctry = pool.country.to_numpy(dtype=object)

    for pi in range(len(pool)):
        c = pool_ctry[pi]
        if c not in s1_by_country:
            # Unseen country fallback: try all countries
            candidates = defaultdict(int)
            for cc, (s1c, idx) in s1_by_country.items():
                for k in _block_keys(pool_nosp[pi], pool_core[pi], pool_addr[pi], pool_nums[pi]):
                    for s1_pos in idx.get(k, []):
                        candidates[s1_pos] += 1
                # Take top candidates
                if candidates:
                    top = sorted(candidates.items(), key=lambda x: -x[1])[:max_cands_per_s1]
                    s1_ids = s1c.entity_id.to_numpy(dtype=object)
                    for s1_pos, cnt in top:
                        results.append((s1_ids[s1_pos], pool_ids[pi], cnt))
            continue

        s1c, idx = s1_by_country[c]
        candidates = defaultdict(int)
        for k in _block_keys(pool_nosp[pi], pool_core[pi], pool_addr[pi], pool_nums[pi]):
            postings = idx.get(k)
            if postings and len(postings) < 10000:  # skip very common keys
                for s1_pos in postings:
                    candidates[s1_pos] += 1

        if candidates:
            top = sorted(candidates.items(), key=lambda x: -x[1])[:max_cands_per_s1]
            s1_ids = s1c.entity_id.to_numpy(dtype=object)
            for s1_pos, cnt in top:
                results.append((s1_ids[s1_pos], pool_ids[pi], cnt))

        if (pi + 1) % 500_000 == 0:
            print(f"    probed {pi+1:,}/{len(pool):,} pool records", flush=True)

    print(f"  probing done in {time.time()-t0:.0f}s | {len(results):,} candidate pairs", flush=True)

    if not results:
        return pd.DataFrame(columns=["s1", "cand", "block_hits"])
    s1_arr, cand_arr, hits_arr = zip(*results)
    return pd.DataFrame({"s1": s1_arr, "cand": cand_arr, "block_hits": hits_arr})

# ─── TF-IDF blocking (for supplementing inverted index) ────────────────────
def block_tfidf(s1, pool, k=3, cache_tag=None):
    """TF-IDF char-ngram blocking per country. Slower but catches fuzzy matches."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    out_parts = []
    for c in pd.unique(pool.country.to_numpy(dtype=object)):
        cache_file = WORK / f"{cache_tag}_{c}.parquet" if cache_tag else None
        if cache_file and cache_file.exists():
            out_parts.append(pd.read_parquet(cache_file))
            print(f"    tfidf {c}: cached", flush=True)
            continue

        t0 = time.time()
        q = pool[pool.country == c]
        s1c = s1[s1.country == c]
        if len(s1c) == 0:
            s1c = s1  # unseen-country fallback

        s1_nosp = s1c.nosp.to_numpy(dtype=object, na_value="")
        q_nosp = q.nosp.to_numpy(dtype=object, na_value="")
        s1_addr = s1c.addr.to_numpy(dtype=object, na_value="")
        q_addr = q.addr.to_numpy(dtype=object, na_value="")

        pairs = defaultdict(lambda: [0.0, 0.0])  # (q_pos, s1_pos) -> [name_sim, addr_sim]

        for field_idx, (s1_text, q_text, kw) in enumerate([
            (s1_nosp, q_nosp, dict(analyzer="char", ngram_range=(3,3))),
            (s1_addr, q_addr, dict(analyzer="char_wb", ngram_range=(3,3)))
        ]):
            vec = TfidfVectorizer(max_df=0.3, sublinear_tf=True, dtype=np.float32, **kw)
            B = vec.fit_transform(s1_text).T.tocsr()
            chunk = 200_000
            for st in range(0, len(q_text), chunk):
                M = sp_matmul_topn(vec.transform(q_text[st:st+chunk]), B,
                                   top_n=k, threshold=0.1, n_threads=os.cpu_count()).tocoo()
                for r, col, v in zip(M.row + st, M.col, M.data):
                    pairs[(r, col)][field_idx] = max(pairs[(r, col)][field_idx], v)

        s1_ids = s1c.entity_id.to_numpy(dtype=object)
        q_ids = q.entity_id.to_numpy(dtype=object)
        rows = []
        for (qi, si), (nsim, asim) in pairs.items():
            rows.append((s1_ids[si], q_ids[qi], nsim, asim))

        res = pd.DataFrame(rows, columns=["s1", "cand", "sim_name", "sim_addr"]) if rows else \
              pd.DataFrame(columns=["s1", "cand", "sim_name", "sim_addr"])
        if cache_file:
            res.to_parquet(cache_file, index=False)
        out_parts.append(res)
        print(f"    tfidf {c}: {len(q):,} queries vs {len(s1c):,} S1 -> {len(res):,} pairs in {time.time()-t0:.0f}s", flush=True)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["s1","cand","sim_name","sim_addr"])

def block_combined(s1, pool, k_tfidf=5, max_inv=20, cache_tag=None):
    """Combine inverted-index and TF-IDF blocking for high recall."""
    cache_file = WORK / f"{cache_tag}_combined.parquet" if cache_tag else None
    if cache_file and cache_file.exists():
        print(f"  combined blocking: cached ({cache_file.name})", flush=True)
        return pd.read_parquet(cache_file)

    print("  Running inverted-index blocking...", flush=True)
    inv = block_inverted(s1, pool, max_cands_per_s1=max_inv)

    print("  Running TF-IDF blocking...", flush=True)
    tfidf = block_tfidf(s1, pool, k=k_tfidf, cache_tag=f"{cache_tag}_tfidf" if cache_tag else None)

    # Merge: union of both blocking methods
    if len(tfidf) > 0:
        inv2 = inv[["s1", "cand"]].assign(src="inv")
        tfidf2 = tfidf[["s1", "cand"]].assign(src="tfidf")
        combined = pd.concat([inv2, tfidf2]).drop_duplicates(subset=["s1","cand"]).reset_index(drop=True)
        # Merge back scores
        combined = combined.merge(inv[["s1","cand","block_hits"]], on=["s1","cand"], how="left")
        combined = combined.merge(tfidf[["s1","cand","sim_name","sim_addr"]], on=["s1","cand"], how="left")
        combined = combined.fillna({"block_hits": 0, "sim_name": 0, "sim_addr": 0})
    else:
        combined = inv.copy()
        combined["sim_name"] = 0.0
        combined["sim_addr"] = 0.0

    combined["score"] = (combined.get("sim_name", 0) + combined.get("sim_addr", 0) * 0.5 +
                         combined.get("block_hits", 0) * 0.1).astype(np.float32)

    if cache_file:
        combined.to_parquet(cache_file, index=False)
    print(f"  combined: {len(combined):,} candidate pairs", flush=True)
    return combined

# ─── Features ───────────────────────────────────────────────────────────────
FEAT_NAMES = [
    "n_tsr", "n_tsort", "n_ratio", "n_part", "n_jw", "n_jac", "n_first", "n_lenr",
    "a_tsr", "a_part", "a_jac", "num_jac", "zip_jac",
    "num_conflict", "zip_conflict",
    "l_addr_missing", "r_addr_missing", "r_nums_missing",
    "block_hits", "sim_name", "sim_addr", "score",
    "n_cands", "is_s3"
]

def _jac(a, b):
    a, b = set(a.split()), set(b.split()); u = a | b
    return len(a & b) / len(u) if u else -1.0

def _first(a, b):
    a, b = a.split(), b.split()
    return float(bool(a) and bool(b) and a[0] == b[0])

def _lenr(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0

def compute_features(cands, s1, pool, step=200_000):
    """Compute pairwise similarity features for candidate pairs."""
    cands = cands.reset_index(drop=True)
    s1_idx = pd.Index(s1.entity_id.to_numpy(dtype=object))
    pool_idx = pd.Index(pool.entity_id.to_numpy(dtype=object))
    lpos = s1_idx.get_indexer(cands.s1.to_numpy(dtype=object))
    rpos = pool_idx.get_indexer(cands.cand.to_numpy(dtype=object))
    assert (lpos >= 0).all() and (rpos >= 0).all(), "IDs missing from s1/pool"

    TXT = ["business_name", "name_full", "core", "nosp", "addr", "nums", "zips"]
    all_feats = []

    for lo in range(0, len(cands), step):
        hi = min(lo + step, len(cands))
        ls, rs = lpos[lo:hi], rpos[lo:hi]
        L = {k: s1[k].iloc[ls].tolist() for k in TXT}
        R = {k: pool[k].iloc[rs].tolist() for k in TXT}
        n = hi - lo

        f = lambda fn, col: np.fromiter((fn(a, b) for a, b in zip(L[col], R[col])), np.float32, n)
        F = {
            "n_tsr": f(fuzz.token_set_ratio, "core"),
            "n_tsort": f(fuzz.token_sort_ratio, "core"),
            "n_ratio": f(fuzz.ratio, "name_full"),
            "n_part": f(fuzz.partial_ratio, "nosp"),
            "n_jw": f(JaroWinkler.similarity, "nosp"),
            "n_jac": f(_jac, "core"),
            "n_first": f(_first, "core"),
            "n_lenr": f(_lenr, "nosp"),
            "a_tsr": f(fuzz.token_set_ratio, "addr"),
            "a_part": f(fuzz.partial_ratio, "addr"),
            "a_jac": f(_jac, "addr"),
            "num_jac": f(_jac, "nums"),
            "zip_jac": f(_jac, "zips"),
        }
        has = lambda side, col: np.fromiter((x != "" for x in side[col]), bool, n)
        ln, rn, lz, rz = has(L, "nums"), has(R, "nums"), has(L, "zips"), has(R, "zips")
        F["num_conflict"] = (ln & rn & (F["num_jac"] == 0)).astype(np.int8)
        F["zip_conflict"] = (lz & rz & (F["zip_jac"] == 0)).astype(np.int8)
        F["l_addr_missing"] = (~has(L, "addr")).astype(np.int8)
        F["r_addr_missing"] = (~has(R, "addr")).astype(np.int8)
        F["r_nums_missing"] = (~rn).astype(np.int8)
        all_feats.append(pd.DataFrame(F))

        if hi % 1_000_000 == 0 or hi == len(cands):
            print(f"    features: {hi:,}/{len(cands):,}", flush=True)

    F = pd.concat(all_feats, ignore_index=True)
    # Add blocking-level features
    for col in ["block_hits", "sim_name", "sim_addr", "score"]:
        if col in cands.columns:
            F[col] = cands[col].to_numpy(np.float32)
        else:
            F[col] = 0.0
    g = cands.groupby("s1")["score"] if "score" in cands.columns else None
    F["n_cands"] = cands.groupby("s1")["cand"].transform("size").to_numpy()
    F["is_s3"] = cands.cand.str.startswith("S3-").to_numpy().astype(np.int8)
    return F

# ─── Decision ───────────────────────────────────────────────────────────────
def decide(cands, probs, all_s1, mode="ef", pmin=0.05, ceil=0.99, thr=0.5):
    d = pd.DataFrame({"s1": cands.s1.to_numpy(), "cand": cands.cand.to_numpy(), "p": probs})
    # one-to-one: each cand maps to highest-prob s1
    d = d.loc[d.groupby("cand")["p"].idxmax()]
    if mode == "thr":
        d = d[d.p >= thr]
    else:
        d = d[d.p >= pmin].sort_values(["s1", "p"], ascending=[True, False])
        g = d.groupby("s1")["p"]
        k = g.cumcount() + 1
        ef = 1.25 * g.cumsum() / (0.25 * g.transform("sum") / ceil + k)
        d = d.assign(k=k, ef=ef)
        best = d.loc[d.groupby("s1")["ef"].idxmax(), ["s1", "k", "ef"]].set_index("s1")
        p0 = np.exp(np.log1p(-d.p.clip(upper=1-1e-6)).groupby(d.s1).sum())
        kmap = best["k"].where(best["ef"] > p0.reindex(best.index), 0)
        d = d[d["k"] <= d["s1"].map(kmap)]
    res = d.groupby("s1")["cand"].agg(list).to_dict()
    return {s: res.get(s, []) for s in all_s1}

def write_tsv(path, col_name, pred_dict, all_s1):
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for s in all_s1:
            f.write(f"{s}\t{','.join(dict.fromkeys(pred_dict.get(s, ())))}\n")

def macro_f05(pred, truth):
    out = []
    for s, t in truth.items():
        p = pred.get(s, set())
        tp = len(p & t)
        if not t:
            out.append(float(not p))
            continue
        if not tp:
            out.append(0.0)
            continue
        P, R = tp / len(p), tp / len(t)
        out.append(1.25 * P * R / (0.25 * P + R))
    return float(np.mean(out))

# ─── Training ───────────────────────────────────────────────────────────────
def train_pipeline(n_s1=150_000):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())

    t_start = time.time()
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    gt = read_gt(); pairs = gt_pairs(gt)
    ALL_COLS = None  # load all columns
    s1 = load("train", 1, ALL_COLS)
    sample = gt.source1_entity_id.sample(n_s1, random_state=42).tolist()
    sp = pairs[pairs.s1.isin(sample)]
    pool = load_pool("train", ALL_COLS)

    # Build query set: true matches + hard negatives + easy negatives
    is_matched = pool.entity_id.isin(pairs.cand)
    is_selected = pool.entity_id.isin(sp.cand)
    q_true = pool[is_selected]
    n_neg = int(0.35 * len(q_true))
    n_hard = n_neg // 2
    q_hard = pool[is_matched & ~is_selected].sample(min(n_hard, (is_matched & ~is_selected).sum()), random_state=42)
    q_easy = pool[~is_matched].sample(n_neg - len(q_hard), random_state=42)
    q = pd.concat([q_true, q_hard, q_easy], ignore_index=True)
    del pool, q_true, q_hard, q_easy, is_matched, is_selected
    print(f"Loaded in {time.time()-t_start:.0f}s | S1: {len(s1):,} | queries: {len(q):,} | true pairs: {len(sp):,}", flush=True)

    # Blocking
    t0 = time.time()
    cands = block_combined(s1, q, k_tfidf=5, max_inv=20, cache_tag=f"train{n_s1}")
    # Deduplicate
    cands = cands.drop_duplicates(subset=["s1", "cand"]).reset_index(drop=True)
    y = cands.merge(pairs.assign(y=1), on=["s1", "cand"], how="left")["y"].fillna(0).to_numpy(np.int8)
    recall = y.sum() / len(sp) if len(sp) > 0 else 0
    print(f"Blocking: {time.time()-t0:.0f}s | candidates: {len(cands):,} | positives: {y.sum():,} | recall: {recall:.4f}", flush=True)

    # Features
    t0 = time.time()
    F = compute_features(cands, s1, q)
    print(f"Features: {time.time()-t0:.0f}s", flush=True)

    # Build ground truth dict
    truth = {s: set() for s in sample}
    for s, m in zip(sp.s1.to_numpy(), sp.cand.to_numpy()):
        truth[s].add(m)

    # Cross-validation
    groups = cands.s1.to_numpy()
    oof = np.zeros(len(cands))
    iters = []
    feat_cols = [c for c in F.columns if c in FEAT_NAMES]

    for fold, (tr, va) in enumerate(GroupKFold(4).split(F, y, groups)):
        m = lgb.train(PARAMS, lgb.Dataset(F[feat_cols].iloc[tr], y[tr]), 3000,
                      valid_sets=[lgb.Dataset(F[feat_cols].iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(F[feat_cols].iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  Fold {fold}: best_iter={m.best_iteration}", flush=True)

    rounds = int(np.mean(iters) * 1.1)

    # Grid search decision params
    def score_pred(cc, p, ids, prm):
        pred = decide(cc, p, ids, **prm)
        return macro_f05({s: set(v) for s, v in pred.items()}, {s: truth[s] for s in ids})

    grid = (
        [dict(mode="ef", pmin=pm, ceil=ce)
         for pm in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
         for ce in (0.9, 0.95, 0.99, 1.0)]
        + [dict(mode="thr", thr=t) for t in np.arange(0.3, 0.95, 0.05)]
    )
    res = sorted(((score_pred(cands, oof, sample, g), g) for g in grid), key=lambda x: -x[0])
    print("\nTop decision configs:")
    for sc, g in res[:8]:
        print(f"  OOF macro F0.5 = {sc:.4f}  {g}")
    best_decision = res[0][1]
    best_score = res[0][0]

    # Train final model on all data
    final = lgb.train(PARAMS, lgb.Dataset(F[feat_cols], y), rounds)
    final.save_model(str(WORK / "lgb.txt"))
    meta = dict(features=feat_cols, decision=best_decision, rounds=rounds,
                k_tfidf=5, max_inv=20, oof_f05=best_score)
    json.dump(meta, open(WORK / "model_meta.json", "w"), indent=2, default=float)

    imp = pd.Series(final.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    print(f"\nTop features (gain):\n{imp.head(15).to_string()}")
    print(f"\nBest OOF F0.5: {best_score:.4f}")
    print(f"Total training time: {time.time()-t_start:.0f}s")
    return best_score

# ─── Prediction ─────────────────────────────────────────────────────────────
def predict_pipeline():
    import lightgbm as lgb

    t_start = time.time()
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)

    meta = json.load(open(WORK / "model_meta.json"))
    model = lgb.Booster(model_file=str(WORK / "lgb.txt"))
    s1 = load("test", 1)
    pool = load_pool("test")
    all_s1 = s1.entity_id.to_numpy(dtype=object).tolist()
    print(f"Loaded in {time.time()-t_start:.0f}s | S1: {len(s1):,} | pool: {len(pool):,}", flush=True)

    # Blocking
    t0 = time.time()
    cands = block_combined(s1, pool, k_tfidf=meta.get("k_tfidf", 5),
                           max_inv=meta.get("max_inv", 20), cache_tag="test")
    cands = cands.drop_duplicates(subset=["s1", "cand"]).reset_index(drop=True)
    print(f"Blocking: {time.time()-t0:.0f}s | candidates: {len(cands):,}", flush=True)

    # Write candidate pairs
    cand_dict = cands.groupby("s1")["cand"].agg(list).to_dict()
    write_tsv(OUT / "candidate_pairs.tsv", "candidate_entity_ids", cand_dict, all_s1)

    # Features
    t0 = time.time()
    F = compute_features(cands, s1, pool)
    feat_cols = meta["features"]
    print(f"Features: {time.time()-t0:.0f}s", flush=True)

    # Predict
    probs = model.predict(F[feat_cols])
    pred = decide(cands, probs, all_s1, **meta["decision"])
    write_tsv(OUT / "matching_results.tsv", "matched_entity_ids", pred, all_s1)

    # Stats
    n = np.array([len(pred[s]) for s in all_s1])
    ctry = s1.country.to_numpy(dtype=object)
    print(f"\nResults: avg matches={n.mean():.2f} | empty={np.mean(n==0):.2%}")
    for c in pd.unique(ctry):
        k = ctry == c
        print(f"  {c}: avg={n[k].mean():.2f} | empty={np.mean(n[k]==0):.2%}")
    print(f"\nTotal prediction time: {time.time()-t_start:.0f}s")

# ─── Eval blocking ─────────────────────────────────────────────────────────
def eval_blocking(n_s1=50_000):
    t0 = time.time()
    gt = read_gt(); pairs = gt_pairs(gt)
    s1 = load("train", 1)
    pool = load_pool("train")

    dev = gt.source1_entity_id.sample(n_s1, random_state=0)
    dev_pairs = pairs[pairs.s1.isin(dev)]
    # Include true matches + some distractors
    q = pd.concat([
        pool[pool.entity_id.isin(dev_pairs.cand)],
        pool[~pool.entity_id.isin(pairs.cand)].sample(int(0.35 * len(dev_pairs)), random_state=0)
    ], ignore_index=True)
    del pool
    print(f"Loaded in {time.time()-t0:.0f}s | queries: {len(q):,}", flush=True)

    cands = block_combined(s1, q, k_tfidf=5, max_inv=20, cache_tag="eval_block")
    cands = cands.drop_duplicates(subset=["s1", "cand"]).reset_index(drop=True)

    # Recall
    tp = cands.merge(dev_pairs, on=["s1", "cand"])
    recall = len(tp) / len(dev_pairs) if len(dev_pairs) > 0 else 0
    avg_cands = cands.groupby("s1").size().reindex(dev, fill_value=0).mean()
    print(f"\nBlocking recall: {recall:.4f} | avg candidates per S1: {avg_cands:.1f}")

    # Show misses
    miss = dev_pairs.merge(cands[["s1","cand"]], on=["s1","cand"], how="left", indicator=True)
    miss = miss[miss._merge == "left_only"][["s1","cand"]]
    print(f"Misses: {len(miss):,} / {len(dev_pairs):,}")
    if len(miss) > 0:
        s1d = s1.set_index("entity_id")
        qd = q.set_index("entity_id")
        for _, (s, m) in miss.head(5).iterrows():
            if s in s1d.index and m in qd.index:
                print(f"  S1:   {s1d.loc[s, 'business_name']} | {s1d.loc[s, 'business_address']}")
                print(f"  MISS: {qd.loc[m, 'business_name']} | {qd.loc[m, 'business_address']}")

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    print("Preprocessing...", flush=True)
    preprocess("train")
    if cmd in ("full", "predict"):
        preprocess("test")

    if cmd == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
        eval_blocking(n)
    elif cmd == "train":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
        train_pipeline(n)
    elif cmd == "predict":
        predict_pipeline()
    elif cmd == "full":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
        train_pipeline(n)
        predict_pipeline()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 pipeline.py [full|train|predict|eval] [N]")
