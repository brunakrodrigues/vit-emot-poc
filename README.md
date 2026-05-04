# vit-emot-poc

Proof of concept for **emotion classification from facial landmarks** using the RAVDESS dataset. Compares three temporal architectures (MLP, CNN1D, Transformer) and includes explainability analysis (XAI). Designed to run on **CPU only**.

---

## Approach

Instead of processing raw video frames, the pipeline extracts **facial landmarks** (2D keypoint coordinates) and treats each clip as a **time series of shape `(T=100, D)`**. This drastically reduces dimensionality and allows lightweight models to run on CPU.

```
RAVDESS video → Facial landmarks (CSV) → Time series T×D → Temporal model → Emotion
```

7 emotion classes (neutral and calm are merged):

`neutral/calm` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

---

## Project Structure

```
vit-emot-poc/
├── data/
│   └── ravdess_landmarks_kaggle/
│       ├── 00_raw_kaggle_csv/      ← Raw CSVs from Kaggle (NOT versioned)
│       ├── 01_processed_T100/      ← Normalized dataset T=100 (.npz)
│       ├── 02_splits/              ← Actor hold-out split (JSON)
│       └── 03_qc/                  ← Manifest and QC report
├── notebooks/
│   ├── 01_ingest_qc_manifest.ipynb
│   ├── 02_preprocess_T100_dataset.ipynb
│   ├── 03_split_actor_holdout.ipynb
│   ├── 04_train_eval_models.ipynb
│   └── 05_xai_attention_deletion.ipynb
├── src/
│   ├── ravdess_utils.py            ← RAVDESS parsing, CSV loading, manifest
│   ├── temporal.py                 ← Temporal normalization (T=100)
│   ├── metrics_utils.py            ← Metrics, seed, bootstrap CI
│   └── models.py                   ← FlatMLP, TemporalCNN1D, EmoTransformer
├── reports/
│   ├── tables/                     ← Consolidated metrics (.csv)
│   └── figures/                    ← Generated plots (.png)
├── runs/
│   └── poc_v1/
│       ├── metrics/                ← Training metrics and XAI scores
│       └── checkpoints/            ← Model checkpoints (.pt)
├── requirements.txt
└── README.md
```

---

## Models

### FlatMLP
Baseline that flattens the full time series before classification (`T × D → Linear`). No notion of temporal order — used as lower-bound reference.

### TemporalCNN1D
1D convolutions along the temporal axis to capture local motion patterns. Faster to train and more parameter-efficient than the MLP baseline.

### EmoTransformer
Transformer encoder with a **CLS token** for classification. Architecture: `d_model=64`, `2 layers`, `4 attention heads`, sinusoidal positional encoding. The CLS token aggregates information from all frames via attention, which also enables attention-based XAI.

---

## XAI — Explainability

### Attention Rollout
Propagates attention weights across all Transformer layers, accumulating residual attention. The CLS token's attention toward each frame produces a **temporal importance map**.

### Gradient × Input
Computes `|∂logit/∂x × x|` per feature, averaged over frames — highlights which **facial landmarks** (coordinates) most influence the prediction.

### Deletion Test (Fidelity)
Progressively masks the top-k most important frames and measures the drop in predicted class probability, compared against a random masking baseline. A faithful explanation should cause a larger drop than random.

---

## Data

Download the RAVDESS facial landmark tracking CSVs from Kaggle and place them in:

```
data/ravdess_landmarks_kaggle/00_raw_kaggle_csv/
```

The internal structure can contain subfolders (e.g. `Actor_01/`, `Actor_02/`) or flat files — notebook 01 performs recursive discovery.

Raw data is **not versioned** (listed in `.gitignore`).

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Register Jupyter kernel (optional)

```bash
python -m ipykernel install --user --name vit-emot-poc
```

---

## Execution Order

Run the notebooks in sequence from the project root:

| # | Notebook | Description | Main Output |
|---|----------|-------------|-------------|
| 1 | `01_ingest_qc_manifest.ipynb` | Ingest and QC raw CSVs | `03_qc/manifest.csv`, `qc_report.csv` |
| 2 | `02_preprocess_T100_dataset.ipynb` | Normalize time series to T=100 frames | `01_processed_T100/dataset_T100.npz` |
| 3 | `03_split_actor_holdout.ipynb` | Actor-based hold-out train/test split | `02_splits/split_actor_holdout.json` |
| 4 | `04_train_eval_models.ipynb` | Train and evaluate all 3 models | `runs/poc_v1/`, `reports/` |
| 5 | `05_xai_attention_deletion.ipynb` | XAI: Attention Rollout + Deletion Test | `reports/figures/xai_*.png` |
