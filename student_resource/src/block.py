import os, re, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import sparse_dot_topn
except ImportError:
    pass
from config import CACHE, K

CH = {"nosp": dict(analyzer="char", ngram_range=(3, 3)),
      "addr": dict(analyzer="char_wb", ngram_range=(3, 3))}

def _backend():
    if os.environ.get("ER_BACKEND"): return os.environ["ER_BACKEND"]
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
BACKEND = _backend()
GPU_MODE = os.environ.get("ER_GPU_MODE", "svd")   # svd (fast, ~3 min) | exact (~3 hours)

def _vec(kw):
    return TfidfVectorizer(max_df=0.3, sublinear_tf=True, dtype=np.float32, **kw)

def _txt(s):
    return s.to_numpy(dtype=object, na_value="")

def _rowdot(A, B):                      # row-wise cosine of L2-normalized sparse rows
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()

# ---------- CPU: exact sparse top-k ----------
def _topk_cpu(s1_text, q_text, k, kw, chunk=50_000, thr=0.1, **kwargs):
    from sparse_dot_topn import sp_matmul_topn
    vec = _vec(kw); B = vec.fit_transform(s1_text).T.tocsr()
    R, C, V = [], [], []
    for st in range(0, len(q_text), chunk):
        M = sp_matmul_topn(vec.transform(q_text[st:st + chunk]), B, top_n=k,
                           threshold=thr, n_threads=os.cpu_count()).tocoo()
        R.append(M.row + st); C.append(M.col); V.append(M.data)
        if len(q_text) > chunk:
            print(f"      chunk [{st}:{min(st+chunk, len(q_text))}/{len(q_text)}]: {len(M.data):,} hits", flush=True)
    if not R:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.concatenate(R), np.concatenate(C), np.concatenate(V)

# ---------- GPU: exact TF-IDF cosine (CSR S1 @ dense query batch) ----------
def _topk_gpu_exact(s1_text, q_text, k, kw, bs=None, qchunk=50_000, thr=0.1, device="cuda:0"):
    import torch
    dev = torch.device(device)
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()
    vec = _vec(kw); X = vec.fit_transform(s1_text).tocsr()          # (n_s1, V), L2-normalized
    n_s1 = X.shape[0]

    # Adaptive safe batch size: ensure intermediate (n_s1 x bs) dense buffer is <= 2.5 GiB
    if bs is None:
        user_bs = os.environ.get("ER_GPU_BS")
        if user_bs:
            bs = int(user_bs)
        else:
            safe_bs = int(2.5 * (1024 ** 3) / (n_s1 * 4 + 1))
            bs = min(1024, max(128, (safe_bs // 64) * 64))

    Xt = torch.sparse_csr_tensor(torch.from_numpy(X.indptr.astype(np.int32)),
                                 torch.from_numpy(X.indices.astype(np.int32)),
                                 torch.from_numpy(X.data.astype(np.float32)),
                                 size=X.shape, device=dev)
    kk = min(k, X.shape[0]); R, C, V = [], [], []
    for st in range(0, len(q_text), qchunk):
        tc = time.time()
        Qx = vec.transform(q_text[st:st + qchunk]).tocsr()
        n_hits_before = sum(len(x) for x in R)
        for b in range(0, Qx.shape[0], bs):
            Qb = torch.from_numpy(np.ascontiguousarray(Qx[b:b + bs].toarray().T)).to(dev)  # (V, bs)
            v, i = (Xt @ Qb).topk(kk, dim=0)                                                   # (kk, bs)
            v, i = v.T.cpu().numpy().ravel(), i.T.cpu().numpy().ravel()
            rows = np.repeat(np.arange(st + b, st + b + Qb.shape[1]), kk)
            keep = v >= thr
            R.append(rows[keep]); C.append(i[keep]); V.append(v[keep])
            del Qb
        if len(q_text) > qchunk:
            n_chunk_hits = sum(len(x) for x in R) - n_hits_before
            print(f"      [{device} bs={bs}] chunk [{st}:{min(st+qchunk, len(q_text))}/{len(q_text)}]: {n_chunk_hits:,} hits in {time.time()-tc:.1f}s", flush=True)
    del Xt
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()
    if not R:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.concatenate(R), np.concatenate(C), np.concatenate(V)

# ---------- GPU: SVD approximate + exact re-rank (fallback) ----------
def _proj(svd, X):
    E = svd.transform(X).astype(np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1
    return E / n

def _topk_gpu_svd(s1_text, q_text, k, kw, dim=256, fit_n=300_000,
                  qchunk=100_000, bs=2048, thr=0.1, device="cuda:0"):
    import torch
    from sklearn.decomposition import TruncatedSVD
    dev = torch.device(device)
    vec = _vec(kw); X = vec.fit_transform(s1_text)
    rng = np.random.default_rng(0)
    samp = X[rng.choice(X.shape[0], min(fit_n, X.shape[0]), replace=False)]
    svd = TruncatedSVD(dim, algorithm="randomized", n_iter=3, random_state=0).fit(samp)
    S = torch.from_numpy(_proj(svd, X)).to(dev, torch.float16)
    kk = min(k, X.shape[0])
    R, C, V = [], [], []
    for st in range(0, len(q_text), qchunk):
        tc = time.time()
        Qx = vec.transform(q_text[st:st + qchunk]); n = Qx.shape[0]
        Q = torch.from_numpy(_proj(svd, Qx)).to(dev, torch.float16)
        n_hits_before = sum(len(x) for x in R)
        for b in range(0, n, bs):
            sub_q = Q[b:b + bs]
            sims = sub_q @ S.T
            v, idx = sims.topk(kk, dim=1)
            v_np, idx_np = v.cpu().numpy(), idx.cpu().numpy()
            rows_b = np.repeat(np.arange(st + b, st + b + sub_q.shape[0]), kk)
            cols_b = idx_np.ravel()
            vals_b = v_np.ravel()
            keep_b = vals_b >= thr
            R.append(rows_b[keep_b]); C.append(cols_b[keep_b]); V.append(vals_b[keep_b])
            del sims, v, idx
        del Q
        if len(q_text) > qchunk:
            n_chunk_hits = sum(len(x) for x in R) - n_hits_before
            print(f"      [{device} svd] chunk [{st}:{min(st+qchunk, len(q_text))}/{len(q_text)}]: {n_chunk_hits:,} hits in {time.time()-tc:.1f}s", flush=True)
    del S
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()
    if not R:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.concatenate(R), np.concatenate(C), np.concatenate(V)

def _pick_topk():
    if BACKEND != "gpu": return _topk_cpu, "cpu"
    return (_topk_gpu_exact, "gpu-exact") if GPU_MODE == "exact" else (_topk_gpu_svd, "gpu-svd")

# ---------- blocking ----------
def block(s1, pool, k=K, cache=None):
    """Pool-side blocking: every (pool record, S1) pair in either channel's top-k, with scores."""
    topk, tag = _pick_topk()
    items = list(CH.items())
    n_gpus = 0
    if BACKEND == "gpu":
        try:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n_gpus = 0

    out = []
    for c in pd.unique(pool["country"].to_numpy(dtype=object)):
        f = CACHE / f"{cache}_{re.sub(r'[^a-z0-9]+', '_', str(c))}.parquet" if cache else None
        if f is not None and f.exists():
            out.append(pd.read_parquet(f)); print(f"  block {c}: cached ({f.name})", flush=True); continue
        t = time.time()
        q = pool[pool["country"] == c]
        idx = s1[s1["country"] == c]
        if len(idx) == 0: idx = s1                              # unseen-country fallback

        if n_gpus >= 2 and BACKEND == "gpu":
            from concurrent.futures import ThreadPoolExecutor
            def _run_channel(ch_idx):
                col, kw = items[ch_idx]
                dev = f"cuda:{ch_idx}"
                tc = time.time()
                r, s, v = topk(_txt(idx[col]), _txt(q[col]), k, kw, device=dev)
                print(f"    {c}/{col} on {dev}: {len(r):,} hits in {time.time()-tc:.0f}s", flush=True)
                return col, pd.DataFrame({"q": r, "s": s, col: v})

            with ThreadPoolExecutor(max_workers=2) as ex:
                results = dict(ex.map(_run_channel, range(2)))
            parts = [results["nosp"], results["addr"]]
        else:
            parts = []
            for col, kw in CH.items():
                tc = time.time()
                r, s, v = topk(_txt(idx[col]), _txt(q[col]), k, kw)
                parts.append(pd.DataFrame({"q": r, "s": s, col: v}))
                print(f"    {c}/{col}: {len(r):,} hits in {time.time()-tc:.0f}s", flush=True)
        m = parts[0].merge(parts[1], on=["q", "s"], how="outer").fillna(0.0)
        m["score"] = m[["nosp", "addr"]].max(axis=1) + 0.5 * m[["nosp", "addr"]].min(axis=1)
        m = m.sort_values(["q", "score"], ascending=[True, False]).reset_index(drop=True)
        m["best"] = m.groupby("q")["score"].transform("first")
        m["rank"] = m.groupby("q").cumcount()
        second = m.loc[m["rank"] == 1].set_index("q")["score"]
        m["gap"] = m["best"] - m["q"].map(second).fillna(0.0)
        s1_ids, q_ids = idx["entity_id"].to_numpy(dtype=object), q["entity_id"].to_numpy(dtype=object)
        res = pd.DataFrame({"s1": s1_ids[m["s"].to_numpy()], "cand": q_ids[m["q"].to_numpy()],
                            "sim_name": m["nosp"].to_numpy(np.float32),
                            "sim_addr": m["addr"].to_numpy(np.float32),
                            "score": m["score"].to_numpy(np.float32),
                            "best": m["best"].to_numpy(np.float32),
                            "rank": m["rank"].to_numpy(np.int16),
                            "gap": m["gap"].to_numpy(np.float32)})
        if f is not None: res.to_parquet(f, index=False)
        out.append(res)
        print(f"  block {c} [{tag}]: {len(q):,} queries vs {len(idx):,} S1 in {time.time()-t:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)

def prune(raw, margin=1.0, floor=0.0, max_rank=10):
    keep = (raw["best"] >= floor) & (raw["score"] >= raw["best"] - margin) & (raw["rank"] < max_rank)
    return raw[keep].reset_index(drop=True)
