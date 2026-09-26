# 🦅 Kaggle GPU Run Guide: Business Entity Resolution

This guide walks you through running the Entity Resolution pipeline on **Kaggle Notebooks** with **30 Free GPU Hours per week** (Dual NVIDIA T4 GPUs).

---

## 🎯 Why Kaggle?
1. **No "Compute Units" Limit:** Kaggle provides 30 hours of free GPU time every single week.
2. **Dual NVIDIA T4 GPUs:** 16 GB VRAM per GPU.
3. **High-Speed Storage:** 30 GB scratch disk at `/kaggle/working`.
4. **1-Click Download:** Outputs generated in `/kaggle/working` appear directly in the **Output** tab on the right panel for immediate download.

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

#### Option B: In a Blank Notebook Cell
Simply paste this in the first cell and run:
```python
# 1. Clone repository
!git clone https://github.com/Luciferxy/amazon-entity-resolution.git /kaggle/working/amazon-entity-resolution
%cd /kaggle/working/amazon-entity-resolution/student_resource

# 2. Install dependencies
!pip install -q pandas pyarrow numpy scipy scikit-learn lightgbm rapidfuzz anyascii sparse_dot_topn lightning

# 3. Configure GPU & Run
import os
os.environ['ER_BACKEND'] = 'gpu'
os.environ['ER_GPU_MODE'] = 'exact'
os.environ['ER_K'] = '5'

# 4. Preprocess & Train
!python src/prep.py train
!python src/prep.py test
!python src/train_lightning.py 100000

# 5. Predict Test Set
!python src/run_test.py

# 6. Copy outputs for 1-click download
!cp output/matching_results.tsv /kaggle/working/matching_results.tsv
!cp output/candidate_pairs.tsv /kaggle/working/candidate_pairs.tsv
!zip -j /kaggle/working/submission_files.zip output/matching_results.tsv output/candidate_pairs.tsv
print("Done! Check the 'Output' tab on the right panel to download your files.")
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
