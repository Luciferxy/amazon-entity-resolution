import numpy as np, pandas as pd

def one_to_one(c, p):
    d = pd.DataFrame({"s1": c["s1"].to_numpy(), "cand": c["cand"].to_numpy(), "p": p})
    return d.loc[d.groupby("cand")["p"].idxmax()]

def decide(c, p, all_s1, mode="ef", pmin=0.05, ceil=0.99, thr=0.5):
    d = one_to_one(c, p)
    if mode == "thr":
        d = d[d.p >= thr]
    else:                                                  # expected-F0.5 set selection
        d = d[d.p >= pmin].sort_values(["s1", "p"], ascending=[True, False])
        g = d.groupby("s1")["p"]
        k = g.cumcount() + 1
        ef = 1.25 * g.cumsum() / (0.25 * g.transform("sum") / ceil + k)
        d = d.assign(k=k, ef=ef)
        best = d.loc[d.groupby("s1")["ef"].idxmax(), ["s1", "k", "ef"]].set_index("s1")
        p0 = np.exp(np.log1p(-d.p.clip(upper=1 - 1e-6)).groupby(d.s1).sum())  # P(no match)
        kmap = best["k"].where(best["ef"] > p0.reindex(best.index), 0)
        d = d[d["k"] <= d["s1"].map(kmap)]
    res = d.groupby("s1")["cand"].agg(list).to_dict()
    return {s: res.get(s, []) for s in all_s1}

def write(path, col, d, all_s1):
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in all_s1:
            f.write(f"{s}\t{','.join(dict.fromkeys(d.get(s, ())))}\n")
