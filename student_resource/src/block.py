import os, re, time, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import sparse_dot_topn
except ImportError:
    pass
from config import CACHE

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
GPU_MODE = os.environ.get("ER_GPU_MODE", "exact")   # exact | svd

def _vec(kw):
    return TfidfVectorizer(max_df=0.3, sublinear_tf=True, dtype=np.float32, **kw)

def _txt(s):
    return s.to_numpy(dtype=object, na_value="")

def _rowdot(A, B):                      # row-wise cosine of L2-normalized sparse rows
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()

# ---------- CPU: exact sparse top-k ----------
def _topk_cpu(s1_text, q_text, k, kw, chunk=50_000, thr=0.1):
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
def _topk_gpu_exact(s1_text, q_text, k, kw, bs=512, qchunk=50_000, thr=0.1):
    import torch
    vec = _vec(kw); X = vec.fit_transform(s1_text).tocsr()          # (n_s1, V), L2-normalized
    Xt = torch.sparse_csr_tensor(torch.from_numpy(X.indptr.astype(np.int32)),
                                 torch.from_numpy(X.indices.astype(np.int32)),
                                 torch.from_numpy(X.data.astype(np.float32)),
                                 size=X.shape, device="cuda")
    kk = min(k, X.shape[0]); R, C, V = [], [], []
    for st in range(0, len(q_text), qchunk):
        Qx = vec.transform(q_text[st:st + qchunk]).tocsr()
        for b in range(0, Qx.shape[0], bs):
            Qb = torch.from_numpy(np.ascontiguousarray(Qx[b:b + bs].toarray().T)).to("cuda")  # (V, bs)
            v, i = (Xt @ Qb).topk(kk, dim=0)                                                   # (kk, bs)
            v, i = v.T.cpu().numpy().ravel(), i.T.cpu().numpy().ravel()
            rows = np.repeat(np.arange(st + b, st + b + Qb.shape[1]), kk)
            keep = v >= thr
            R.append(rows[keep]); C.append(i[keep]); V.append(v[keep])
            del Qb
    del Xt; torch.cuda.empty_cache()
    return np.concatenate(R), np.concatenate(C), np.concatenate(V)

# ---------- GPU: SVD approximate + exact re-rank (fallback) ----------
def _proj(svd, X):
    E = svd.transform(X).astype(np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1
    return E / n

def _topk_gpu_svd(s1_text, q_text, k, kw, dim=256, k_ann=10, fit_n=300_000,
                  qchunk=100_000, bs=1024, thr=0.1):
    import torch
    from sklearn.decomposition import TruncatedSVD
    vec = _vec(kw); X = vec.fit_transform(s1_text)
    rng = np.random.default_rng(0)
    samp = X[rng.choice(X.shape[0], min(fit_n, X.shape[0]), replace=False)]
    svd = TruncatedSVD(dim, algorithm="randomized", n_iter=4, random_state=0).fit(samp)
    S = torch.from_numpy(_proj(svd, X)).to("cuda", torch.float16)
    ka = min(k_ann, X.shape[0]); kk = min(k, ka)
    R, C, V = [], [], []
    for st in range(0, len(q_text), qchunk):
        Qx = vec.transform(q_text[st:st + qchunk]); n = Qx.shape[0]
        Q = torch.from_numpy(_proj(svd, Qx)).to("cuda", torch.float16)
        I = np.empty((n, ka), dtype=np.int64)
        for b in range(0, n, bs):
            I[b:b + bs] = (Q[b:b + bs] @ S.T).topk(ka, dim=1).indices.cpu().numpy()
        ex = _rowdot(Qx[np.repeat(np.arange(n), ka)], X[I.ravel()]).reshape(n, ka)
        order = np.argsort(-ex, axis=1)[:, :kk]
        cols = np.take_along_axis(I, order, 1).ravel()
        vals = np.take_along_axis(ex, order, 1).ravel()
        rows = np.repeat(np.arange(n) + st, kk)
        keep = vals >= thr
        R.append(rows[keep]); C.append(cols[keep]); V.append(vals[keep])
        del Q
    del S; torch.cuda.empty_cache()
    return np.concatenate(R), np.concatenate(C), np.concatenate(V)

def _pick_topk():
    if BACKEND != "gpu": return _topk_cpu, "cpu"
    return (_topk_gpu_exact, "gpu-exact") if GPU_MODE == "exact" else (_topk_gpu_svd, "gpu-svd")

# ---------- blocking ----------
def block(s1, pool, k=3, cache=None):
    """Pool-side blocking: every (pool record, S1) pair in either channel's top-k, with scores."""
    topk, tag = _pick_topk()
    out = []
    for c in pd.unique(pool["country"].to_numpy(dtype=object)):
        f = CACHE / f"{cache}_{re.sub(r'[^a-z0-9]+', '_', str(c))}.parquet" if cache else None
        if f is not None and f.exists():
            out.append(pd.read_parquet(f)); print(f"  block {c}: cached ({f.name})", flush=True); continue
        t = time.time()
        q = pool[pool["country"] == c]
        idx = s1[s1["country"] == c]
        if len(idx) == 0: idx = s1                              # unseen-country fallback
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
