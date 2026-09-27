# 🦅 Kaggle GPU Run Guide: Business Entity Resolution

This guide walks through the Kaggle workflow. Candidate generation combines name and address character and word n-grams; LightGBM then scores the retrieved pairs.

---

## 🎯 Why Kaggle?
1. **GPU support:** The notebook uses both T4 GPUs when Kaggle provides two; a single GPU is supported.
2. **High-Speed Storage:** Intermediate files use the Kaggle workspace.
3. **Downloadable outputs:** Copy the final TSV files and archive to `/kaggle/working`.

---

## 🛠️ Step-by-Step Instructions

### Step 1: Create a Kaggle Notebook
1. Go to **[kaggle.com/code](https://www.kaggle.com/code)** and sign in.
2. Click **"New Notebook"**.

---

### Step 2: Configure Notebook Settings (Important!)
In the panel on the **right-hand side** under **Notebook options**:
1. **Accelerator:** Change from *None* to **GPU T4 x2** (or *GPU T4*).
2. **Internet:** Toggle to **ON** (required to clone GitHub and install packages).

---

### Step 3: Upload the Dataset to Kaggle
1. In the top-right menu of the notebook, click **+ Add Input**.
2. Click **Upload Dataset** (top right of the popup).
3. Title it: `amazon-er-dataset`.
4. Drag and drop your local `dataset/` folder (or `dataset.zip`).
5. Click **Create**.
*Kaggle will mount it at `/kaggle/input/amazon-er-dataset/...` (our code auto-detects it!).*

---

### Step 4: Run the Notebook

#### Option A: Import the Kaggle Runbook (Easiest)
In Kaggle's top menu:
1. Click **File** $\to$ **Import Notebook**.
2. Upload [`kaggle_runbook.ipynb`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/kaggle_runbook.ipynb) from your computer (or select GitHub import).
3. Click **Run All**!

The notebook's first cell clones the GitHub repository. For these local improvements to be used, that clone must contain the updated `src/` files; push the changes to the repo before running, or upload the updated source files to Kaggle and use that copy.

#### Option B: In a Blank Notebook Cell
Simply paste this in the first cell and run:
```python
# 1. Clone or pull latest repository
import os
if not os.path.exists('/kaggle/working/amazon-entity-resolution'):
    !git clone https://github.com/Luciferxy/amazon-entity-resolution.git /kaggle/working/amazon-entity-resolution
else:
    %cd /kaggle/working/amazon-entity-resolution
    !git pull origin main

%cd /kaggle/working/amazon-entity-resolution/student_resource

# 2. Install dependencies
!pip install -q pandas pyarrow numpy scipy scikit-learn lightgbm rapidfuzz anyascii sparse_dot_topn transformers

# 3. Configure GPU and Memory-Safe Environment
import os, torch
os.environ['ER_BACKEND'] = 'gpu' if torch.cuda.is_available() else 'cpu'
os.environ['ER_GPU_MODE'] = 'svd'
os.environ['ER_K'] = '5'
os.environ['ER_FEAT_JOBS'] = '1'       # Prevents memory spikes in Kaggle
os.environ['ER_PREP_WORKERS'] = '2'    # Safe preprocessing memory
os.environ['CE_BATCH_SIZE'] = '64'
os.environ['CE_EPOCHS'] = '2'
os.environ['ER_CE_MODE'] = 'cascade'   # Cross-Encoder cascade on borderline pairs

# 4. Fine-Tune Cross-Encoder on Hard Pairs (~5-6 min -> Validation F0.5 = 0.9951)
if not os.path.exists("work/cross_encoder_finetuned"):
    print("\n--- Fine-Tuning Cross-Encoder on Hard Pairs ---")
    !python -u src/train_cross_encoder.py
else:
    print("\n--- Using existing fine-tuned Cross-Encoder in work/cross_encoder_finetuned ---")

# 5. Run test inference with Cascade Cross-Encoder
!python -u src/run_test.py

# 6. Copy outputs for 1-click download
!cp output/matching_results.tsv /kaggle/working/matching_results.tsv
!cp output/candidate_pairs.tsv /kaggle/working/candidate_pairs.tsv
!zip -j /kaggle/working/submission_files.zip output/matching_results.tsv output/candidate_pairs.tsv
print("\nSUCCESS! Check the 'Output' tab on the right panel to download matching_results.tsv and submission_files.zip.")
```

---

### 📦 Downloading Your Submission Files
When the run finishes:
1. Look at the **right-hand panel** under **Output**.
2. Refresh the section: you will see:
   - `submission_files.zip`
   - `matching_results.tsv`
   - `candidate_pairs.tsv`
3. Click the three dots ($\dots$) next to `submission_files.zip` $\to$ **Download**.
