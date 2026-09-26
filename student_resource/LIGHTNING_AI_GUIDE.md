# ⚡ Lightning AI Guide: Business Entity Resolution

This guide details how to run the Entity Resolution pipeline on **Lightning AI Studio** (https://lightning.ai) to take full advantage of cloud NVIDIA GPUs (T4 / A10G / L4 / A100) for **10–20x faster candidate blocking** and training on larger entity samples.

---

## 🚀 Why Use Lightning AI for this Project?

1. **GPU-Accelerated Blocking (`Xt @ Qb`):**
   - On local Mac CPU, exact TF-IDF blocking across 1.73M queries and 9.97M candidates takes ~3 hours.
   - On an NVIDIA GPU (T4/A10G) in Lightning AI, PyTorch CUDA sparse CSR tensor multiplication computes top-5 candidates in **~10–15 minutes**.
2. **Larger Scale Training:**
   - Instead of small samples (10k–50k), you can comfortably train on **300,000 to 500,000+ S1 entities** with 28 engineered tabular features.
3. **Dual Model Support:**
   - Fast gradient-boosted trees via **LightGBM**.
   - Optional Deep Residual Neural Matcher built with **PyTorch Lightning** (`src/lightning_matcher.py`).
4. **Guaranteed Competition Compliance:**
   - Outputs both required files: `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
   - Strictly validates against `utils/validate_submission.py`.

---

## 🛠️ Step-by-Step Setup on Lightning AI

### Step 1: Create a Lightning Studio
1. Log in to [Lightning AI](https://lightning.ai).
2. Click **"New Studio"** (or open an existing one).
3. Under Environment/Hardware, select:
   - **T4 GPU** (available on free tier / community credits), or
   - **A10G / L4 GPU** (for even faster memory bandwidth).

### Step 2: Add Project Files to the Studio
You can upload the project using one of three convenient methods:

#### Option A: Drag-and-Drop in Lightning Studio
- In the Studio file explorer on the left, simply drag and drop the `student_resource` directory into the workspace.

#### Option B: Clone via Git
If you have pushed your project to GitHub or GitLab:
```bash
git clone <your-repo-url>
cd student_resource
```

#### Option C: Upload Archive
On your local machine, zip the project (excluding large caches if you want a fast upload):
```bash
zip -r student_resource.zip student_resource/ -x "student_resource/work/*.parquet"
```
Upload `student_resource.zip` into the Studio and extract:
```bash
unzip student_resource.zip
cd student_resource
```

---

## 🎯 Running the Pipeline on Lightning AI

Choose the workflow that suits your preference:

### Workflow 1: Interactive Jupyter Notebook (Recommended)
Open [`lightning_studio.ipynb`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/lightning_studio.ipynb) directly in Lightning Studio:
1. **Cell 1**: Verifies CUDA GPU detection and available VRAM.
2. **Cell 2**: Installs dependencies (`lightning`, `lightgbm`, `rapidfuzz`, `sparse_dot_topn`, etc.).
3. **Cell 3**: Sets `ER_BACKEND="gpu"` and `ER_K="5"`.
4. **Cell 4–5**: Preprocessing & Candidate generation recall check ($K=5$).
5. **Cell 6**: Trains the model on 300,000 entities with 4-fold GroupKFold.
6. **Cell 7**: Runs test inference and writes `candidate_pairs.tsv` and `matching_results.tsv`.
7. **Cell 8**: Automatically runs `validate_submission.py`.
8. **Cell 9**: Zips submission files to `output/submission_files.zip`.

---

### Workflow 2: One-Click Terminal Execution
In the Lightning Studio terminal:
```bash
bash run_lightning.sh
```
This single command handles everything automatically from dependency installation to test inference and validation.

To run with a custom training sample size (e.g. 500,000 S1 entities):
```bash
bash run_lightning.sh 500000
```

---

### Workflow 3: Custom Python CLI
To train with both LightGBM and the PyTorch Lightning Neural Matcher:
```bash
# Set GPU mode
export ER_BACKEND=gpu
export ER_GPU_MODE=exact
export ER_K=5

# Train with PyTorch Lightning module enabled
python3 src/train_lightning.py 300000 --use-nn

# Run test prediction
python3 src/run_test.py
```

---

## 📊 File Architecture

| File | Purpose |
| :--- | :--- |
| [`lightning_studio.ipynb`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/lightning_studio.ipynb) | Interactive notebook for Lightning AI Studio |
| [`run_lightning.sh`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/run_lightning.sh) | Turnkey bash script for 1-click cloud execution |
| [`src/train_lightning.py`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/src/train_lightning.py) | GPU-enabled training pipeline with GroupKFold and LOCO |
| [`src/lightning_matcher.py`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/src/lightning_matcher.py) | PyTorch Lightning Deep Residual Tabular Matcher |
| [`src/block.py`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/src/block.py) | GPU sparse CSR cosine top-$k$ candidate generation |
| [`src/features.py`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/src/features.py) | 28 dense string, token, and numeric features |
| [`utils/validate_submission.py`](file:///Users/souravsuman/Documents/ChatGPT/Amazon/student_resource/utils/validate_submission.py) | Official format validator |

---

## 🏆 Submission Deliverables
Upon completion, the following two files are produced under `output/`:
- `output/candidate_pairs.tsv` (Source 1 ID to candidate entity IDs, avg ~3.3 candidates per entity)
- `output/matching_results.tsv` (Source 1 ID to matched entity IDs)
- `output/submission_files.zip` (Ready to submit to the hackathon portal)
