# 🚀 Google Colab Run Guide: Business Entity Resolution

This guide explains how to run the entire pipeline on **Google Colab** using a free **NVIDIA T4 GPU** and **Google Drive**.

---

## 🎯 Why Google Colab?

1. **Free NVIDIA T4 GPU (15 GB VRAM):**
   - Candidate blocking across 1.73M queries and 9.97M pool entities takes **~10–12 minutes** on GPU instead of ~3 hours on local CPU.
2. **No GitHub File Size Limits:**
   - The 2.3 GB dataset and preprocessed Parquet files live directly on your Google Drive.
3. **Resilient Outputs:**
   - Preprocessed Parquets and final submission TSVs are saved directly on Google Drive, so nothing is lost if your browser tab closes or disconnects.

---

## 🛠️ Step-by-Step Instructions

### Step 1: Upload Project to Google Drive
1. Open [Google Drive](https://drive.google.com).
2. Upload the [`student_resource`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource) folder into your Drive (e.g. into `MyDrive/student_resource`).
   - *Ensure it contains `dataset/` (the raw TSVs), `src/`, `utils/`, and `colab_runbook.ipynb`.*

---

### Step 2: Open Notebook in Google Colab
You have two convenient options:

#### Option A: Direct from GitHub
Click the Colab badge:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Luciferxy/amazon-entity-resolution/blob/main/student_resource/colab_runbook.ipynb)

#### Option B: From Google Drive
1. In Google Drive, navigate to the uploaded `student_resource` folder.
2. Right-click on [`colab_runbook.ipynb`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/colab_runbook.ipynb) → **Open with** → **Google Colaboratory**.

---

### Step 3: Enable T4 GPU
In Colab:
1. Click **Runtime** in the top menu bar.
2. Select **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.

---

### Step 4: Run the Cells
Simply click **Runtime → Run all** (or execute cell-by-cell):

| Cell | Purpose | Expected Duration |
| :--- | :--- | :--- |
| **Cell 1** | Mounts Google Drive & auto-locates `student_resource` | ~15 sec |
| **Cell 2** | Verifies NVIDIA T4 GPU and VRAM | ~2 sec |
| **Cell 3** | Installs dependencies (`lightgbm`, `rapidfuzz`, `sparse_dot_topn`, `lightning`) | ~20 sec |
| **Cell 4** | Configures GPU backend (`ER_BACKEND=gpu`, `ER_K=5`) | Instant |
| **Cell 5** | Preprocesses raw TSVs into Parquets under `work/` | ~3 min |
| **Cell 6** | Evaluates candidate generation recall ($K=5$) | ~2 min |
| **Cell 7** | Trains model on 300,000 S1 entities | ~3–4 min |
| **Cell 8** | Runs full test set inference and generates candidate pairs & matches | ~10–12 min |
| **Cell 9** | Validates submission format with `utils/validate_submission.py` | ~1 min |
| **Cell 10** | Zips final submission files to `output/submission_files.zip` | ~10 sec |

**Total Runtime:** ~20–25 minutes!

---

### 📦 Output Files
When completed, the following files are saved in your Google Drive under `student_resource/output/`:
- `candidate_pairs.tsv` (~3.3 candidates per S1 entity)
- `matching_results.tsv` (Optimized predictions for $F_{0.5} \ge 0.98$)
- `submission_files.zip` (Download and submit directly to the hackathon portal)
