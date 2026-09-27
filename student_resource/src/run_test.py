try:
    import sparse_dot_topn
except ImportError:
    pass
import os, gc, json, time, numpy as np, pandas as pd, lightgbm as lgb
from config import W, OUT, CACHE, ALL_COLS, DATASET, load, load_pool
from prep import get_tsv_path
from block import block_country, prune
from features import features
from decide import decide, write

def main():
    t0 = time.time()
    meta = json.load(open(W / "model_meta.json"))
    model = lgb.Booster(model_file=str(W / "lgb.txt"))
    feature_names = meta["features"]
    k_cand = meta.get("K", 5)
    prune_params = meta.get("prune", {})
    decision_params = meta.get("decision", {})

    # Ensure test parquet files exist
    if not (W / "test_s1.parquet").exists():
        from prep import prep
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

    # Load S1 entity ID ordering and country mapping lightly (2 columns only!)
    s1_meta = load("test", 1, ["entity_id", "country"])
    all_s1 = s1_meta["entity_id"].to_numpy(dtype=object).tolist()
    ctry_arr = s1_meta["country"].to_numpy(dtype=object)
    countries = list(pd.unique(ctry_arr))
    del s1_meta
    gc.collect()

    # Check Cross-Encoder mode
    ce_mode = os.environ.get("ER_CE_MODE", "cascade")
    load_cols = ALL_COLS + (["business_address"] if ce_mode in ("full", "cascade") else [])

    # Initialize shared Cross-Encoder engine if needed
    ce_engine = None
    if ce_mode in ("full", "cascade"):
        from hybrid_cross_encoder import FastCrossEncoder, DEFAULT_MODEL
        finetuned_dir = W / "cross_encoder_finetuned"
        model_path = str(finetuned_dir) if finetuned_dir.exists() else DEFAULT_MODEL
        print(f"Initializing FastCrossEncoder ({model_path})...", flush=True)
        ce_engine = FastCrossEncoder(model_name=model_path)

    print(f"\nProcessing countries sequentially: {countries} (Total S1 entities: {len(all_s1):,})", flush=True)

    all_matches = {}
    all_candidates = {}

    for ctry in countries:
        t_c = time.time()
        print(f"\n=================== Country: {ctry} ===================", flush=True)

        # Load ONLY this country's data directly from disk using pushdown filters!
        s1_c = load("test", 1, load_cols, country=ctry)
        pool_c = load_pool("test", load_cols, country=ctry)
        s1_c_ids = s1_c["entity_id"].to_numpy(dtype=object).tolist()
        print(f"  [{ctry}] Loaded: S1 {len(s1_c):,} | Pool {len(pool_c):,}", flush=True)

        # 1. Blocking per country (reads from disk cache if exists)
        c_c = block_country(s1_c, pool_c, ctry, k=k_cand, cache="test_raw")
        if prune_params:
            c_c = prune(c_c, **prune_params)
        print(f"  [{ctry}] Candidate pairs: {len(c_c):,}", flush=True)

        # Record candidates for candidate_pairs.tsv
        cand_map = c_c.groupby("s1")["cand"].agg(list).to_dict()
        all_candidates.update(cand_map)
        del cand_map

        # 2. Vectorized Feature Extraction & LightGBM Prediction in memory-safe chunks
        g = c_c.groupby("s1")["score"]
        c_c["n_cands"] = g.transform("size").to_numpy(dtype=np.int32)
        c_c["score_vs_best_s1"] = (c_c["score"] - g.transform("max")).to_numpy(dtype=np.float32)
        c_c["is_s3"] = c_c["cand"].str.startswith("S3-").to_numpy(dtype=np.int8)

        CHUNK_SIZE = 2_000_000
        n_chunks = (len(c_c) + CHUNK_SIZE - 1) // CHUNK_SIZE
        probs_list = []
        for ch_idx in range(n_chunks):
            start_idx = ch_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, len(c_c))
            sub_c = c_c.iloc[start_idx:end_idx].reset_index(drop=True)
            t_f = time.time()
            F_sub = features(sub_c, s1_c, pool_c)
            p_sub = model.predict(F_sub[feature_names])
            probs_list.append(p_sub.astype(np.float32))
            print(f"  [{ctry}] Chunk {ch_idx+1}/{n_chunks} ({len(sub_c):,} pairs): features+LGB in {time.time()-t_f:.1f}s", flush=True)
            del sub_c, F_sub
            gc.collect()

        p_c = np.concatenate(probs_list) if probs_list else np.empty(0, dtype=np.float32)
        del probs_list
        gc.collect()

        # 3. Cross-Encoder Cascade Rescoring on borderline pairs
        if ce_mode == "cascade" and ce_engine is not None:
            from hybrid_cross_encoder import cascade_rescore
            print(f"  [{ctry}] Running Cascade Cross-Encoder Rescoring...", flush=True)
            p_c = cascade_rescore(c_c, p_c, s1_c, pool_c, low_thr=0.15, high_thr=0.70, ce_weight=0.50, ce_engine=ce_engine)
        elif ce_mode == "full":
            from hybrid_cross_encoder import full_cross_encoder_score
            print(f"  [{ctry}] Running Full Cross-Encoder Scoring...", flush=True)
            ce_p = full_cross_encoder_score(c_c, s1_c, pool_c, batch_size=512, max_length=96)
            p_c = 0.40 * p_c + 0.60 * ce_p

        # 4. Decision Rule
        pred_c = decide(c_c, p_c, s1_c_ids, **decision_params)
        all_matches.update(pred_c)
        n_matched = sum(len(v) for v in pred_c.values())
        print(f"  [{ctry}] Finished in {time.time()-t_c:.1f}s | Matches: {n_matched:,}", flush=True)

        del c_c, p_c, s1_c, pool_c
        gc.collect()

    # 5. Write submission files
    print("\nWriting output/candidate_pairs.tsv...", flush=True)
    write(OUT / "candidate_pairs.tsv", "candidate_entity_ids", all_candidates, all_s1)

    print("Writing output/matching_results.tsv...", flush=True)
    write(OUT / "matching_results.tsv", "matched_entity_ids", all_matches, all_s1)

    n = np.array([len(all_matches.get(s, [])) for s in all_s1])
    print(f"\nFinal Stats: avg matches {n.mean():.2f} | empty {np.mean(n == 0):.2%}")
    for A in pd.unique(ctry_arr):
        k = ctry_arr == A
        print(f"  {A}: avg matches {n[k].mean():.2f} | empty {np.mean(n[k] == 0):.2%}")

    print("\nRunning submission validator...", flush=True)
    import subprocess, sys
    test_dir = str(get_tsv_path("test", 1).parent)
    val_res = subprocess.run([
        sys.executable, str(W.parent / "utils/validate_submission.py"),
        "--matching", str(OUT / "matching_results.tsv"),
        "--candidate", str(OUT / "candidate_pairs.tsv"),
        "--test-dir", test_dir
    ])
    print(f"Validator exit code: {val_res.returncode}", flush=True)

if __name__ == "__main__":
    main()
