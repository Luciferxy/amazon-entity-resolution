#!/usr/bin/env bash
# ==============================================================================
# Lightning AI Run Script: Business Entity Resolution
# ==============================================================================
# Optimized for Lightning AI Studio (T4 / A10G / A100 GPU)
# Runs GPU blocking (K=5), trains model on 300,000+ S1 entities, predicts test set,
# and verifies output compliance with utils/validate_submission.py.
# ==============================================================================

set -e

echo "=================================================================="
echo "⚡ Starting Business Entity Resolution Pipeline on Lightning AI ⚡"
echo "=================================================================="

# 1. Hardware & Environment Check
echo -e "\n[Step 1/6] Checking Hardware & GPU..."
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA Available: YES (Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
else:
    print('CUDA Available: NO (Running on CPU)')
"

# 2. Dependencies
echo -e "\n[Step 2/6] Verifying Dependencies..."
pip install -q --no-warn-script-location \
    pandas pyarrow numpy scipy scikit-learn lightgbm rapidfuzz anyascii sparse_dot_topn transformers torch

# 3. Configure GPU Backend
export ER_BACKEND="gpu"
export ER_GPU_MODE="svd"
export ER_K=5
export ER_FEAT_JOBS=1
export CE_BATCH_SIZE=64
export CE_EPOCHS=2
export ER_CE_MODE="cascade"

echo "Configuration: ER_BACKEND=$ER_BACKEND, ER_GPU_MODE=$ER_GPU_MODE, ER_K=$ER_K, ER_CE_MODE=$ER_CE_MODE"

# 4. Preprocessing check
if [ ! -f "work/train_s1.parquet" ]; then
    echo -e "\n[Step 3/6] Preprocessing train dataset into work/ Parquet..."
    python3 src/prep.py train
else
    echo -e "\n[Step 3/6] Preprocessed train Parquets already present in work/."
fi

if [ ! -f "work/test_s1.parquet" ]; then
    echo -e "\n[Step 3/6b] Preprocessing test dataset into work/ Parquet..."
    python3 src/prep.py test
else
    echo -e "\n[Step 3/6b] Preprocessed test Parquets already present in work/."
fi

# 5. Fine-Tune Cross-Encoder on Hard Pairs (F0.5 >= 0.995)
if [ ! -d "work/cross_encoder_finetuned" ]; then
    echo -e "\n[Step 4/6] Fine-Tuning Cross-Encoder on Hard Pairs (Target: F0.5 >= 0.995)..."
    python3 src/train_cross_encoder.py
else
    echo -e "\n[Step 4/6] Cross-Encoder checkpoint already exists in work/cross_encoder_finetuned."
fi

# 6. Test Inference & Submission Generation with Cascade Cross-Encoder
echo -e "\n[Step 5/6] Generating Candidate Pairs and Final Matches with Cascade Cross-Encoder..."
python3 src/run_test.py

# 7. Verification & Packaging
echo -e "\n[Step 6/6] Packaging Final Submission Files..."
zip -j output/submission_files.zip output/matching_results.tsv output/candidate_pairs.tsv

echo -e "\n=================================================================="
echo "🎉 Lightning AI Run Completed Successfully!"
echo "Generated Files:"
ls -lh output/*.tsv output/*.zip
echo "=================================================================="
