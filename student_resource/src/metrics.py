import numpy as np

def macro_f05(pred, truth):  # dicts: s1 -> set
    out = []
    for s, t in truth.items():
        p = pred.get(s, set()); tp = len(p & t)
        if not t: out.append(float(not p)); continue
        if not tp: out.append(0.0); continue
        P, R = tp / len(p), tp / len(t); out.append(1.25 * P * R / (0.25 * P + R))
    return float(np.mean(out))
