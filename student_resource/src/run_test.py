try:
    import sparse_dot_topn
except ImportError:
    pass
import json, time, numpy as np, pandas as pd, lightgbm as lgb
from config import W, OUT, CACHE, ALL_COLS, DATASET, load, load_pool
from block import block, prune
from features import features
from decide import decide, write

def main():
    t0 = time.time()
    meta = json.load(open(W / "model_meta.json"))
    model = lgb.Booster(model_file=str(W / "lgb.txt"))
    # Ensure test parquet files exist
    if not (W / "test_s1.parquet").exists():
        from prep import prep, get_tsv_path
        if get_tsv_path("test", 1).exists():
            print("Preparing test splits...", flush=True)
            for i in (1, 2, 3):
                prep("test", i)
        else:
            print("\n" + "="*70)
            print("[INFO] Test dataset files (test_source1.tsv, etc.) are not yet attached.")
            print("Your model training and 4-fold CV have completed successfully.")
            print("To generate final competition submission files:")
            print("1. Upload/attach the test dataset folder to this Kaggle notebook.")
            print("2. Re-run: !python src/run_test.py")
            print("="*70 + "\n")
            return

    s1 = load("test", 1, ALL_COLS); pool = load_pool("test", ALL_COLS)
    all_s1 = s1.entity_id.to_numpy(dtype=object).tolist()
    print(f"loaded {time.time()-t0:.0f}s | S1 {len(s1):,} | pool {len(pool):,}", flush=True)

    c = prune(block(s1, pool, meta["K"], cache="test_raw"), **meta["prune"])
    write(OUT / "candidate_pairs.tsv", "candidate_entity_ids",
          c.groupby("s1")["cand"].agg(list).to_dict(), all_s1)
    print(f"candidates {len(c):,} | avg per S1 {len(c)/len(all_s1):.2f}", flush=True)

    t0 = time.time(); F = features(c, s1, pool); print(f"features {time.time()-t0:.0f}s", flush=True)
    p = model.predict(F[meta["features"]])
    c.assign(p=p).to_parquet(CACHE / "test_scored.parquet", index=False)   # for later ensembling

    pred = decide(c, p, all_s1, **meta["decision"])
    write(OUT / "matching_results.tsv", "matched_entity_ids", pred, all_s1)
    n = np.array([len(pred[s]) for s in all_s1]); ctry = s1.country.to_numpy(dtype=object)
    print(f"avg matches {n.mean():.2f} | empty {np.mean(n == 0):.2%}")
    for A in pd.unique(ctry):
        k = ctry == A
        print(f"  {A}: avg matches {n[k].mean():.2f} | empty {np.mean(n[k] == 0):.2%}")

    print("\nRunning submission validator...", flush=True)
    import subprocess, sys
    val_res = subprocess.run([
        sys.executable, str(W.parent / "utils/validate_submission.py"),
        "--matching", str(OUT / "matching_results.tsv"),
        "--candidate", str(OUT / "candidate_pairs.tsv"),
        "--test-dir", str(DATASET / "test")
    ])
    print(f"Validator exit code: {val_res.returncode}", flush=True)

if __name__ == "__main__":
    main()
