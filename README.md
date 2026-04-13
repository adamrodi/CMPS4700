# CMPS 4700 Stellar Classification Project

Group B — Adam Rodi & Ron Logarbo

## Quick Start

### 1. Clone + setup the venv**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the preprocessing
```bash
python3 code/preprocess.py
```

That's it. This will generate the PA1 outputs including two Excel files (raw and preprocessed data) and three PNG plots. Everything goes into `code/output/`.

## What Is This?

We're building a stellar classification system using machine learning. The dataset is from the Sloan Digital Sky Survey (SDSS DR17) and contains ~100k observations of astronomical objects classified as STAR, GALAXY, or QSO (quasar).

**The goal:** Use ANN, SVM, Decision Tree, and K-NN to predict which class an object belongs to, based on photometric and positional features. Pretty standard ML pipeline stuff.

**The assignment:** CMPS 4700 project, broken into phases:

- **PA1:** Preprocessing (raw data cleaning, splitting, scaling, and exploratory plots)

## What's in Here

```text
code/
  preprocess.py          — Preprocessing pipeline (PA1 + extensible)
  input/
    star_classification.csv  — Raw 100k row SDSS dataset
    train/               — Will hold train split later
    test/                — Will hold test split later
  output/
    raw_data.xlsx        — Raw data export for report
    preprocessed_data.xlsx — Cleaned + scaled + split data
    plot_*.png           — Plots for the report
    preprocessing_summary.json — Data about what we did
  model/
    scaler_params.json   — Z-score stats (means/stds) to use on new data
docs/
  Project_4570.pdf       — The actual assignment spec (read this if confused)
  Project_4570.pptx      — Report template
  GroupB_proposal.pptx   — Our proposal to the professor
```

## Environment

We're using a virtual environment to keep python clean. Activate it before running anything:

```bash
source venv/bin/activate
```

If you add new dependencies, do it with pip inside the venv, then update requirements.txt:

```bash
pip install <package>
pip freeze > requirements.txt
```

Then commit requirements.txt.

## Current Status (PA1)

**Done:**

- ✅ Read raw CSV (100k × 18 columns)
- ✅ Dropped 9 identifier columns (obj_ID, run_ID, etc.)
- ✅ Kept 8 features: u, g, r, i, z (photometric filters), alpha, delta (coordinates), redshift
- ✅ Created stratified 60/20/20 train/val/test split (class-balanced)
- ✅ Applied z-score scaling (fit on train only, to avoid leakage)
- ✅ Generated three plots showing data distribution and preprocessing effects
- ✅ Exported two Excel files and metadata JSON

**Verified:**

- Split is perfectly stratified (GALAXY/STAR/QSO proportions maintained)
- Raw: 100k rows, 18 columns
- Preprocessed: 100k rows, 8 features + class + split label
- No data leakage in scaling

**Not done (PA1):**

- PDF report still needs a narrative + embedded plots. Code, data, and plots are ready.

## How the Preprocessing Works

1. **Read** the raw CSV
2. **Clean** by dropping identifier columns we don't need
3. **Check** that the data looks right (schema validation)
4. **Split** into train/val/test using stratified sampling (preserves class balance)
5. **Scale** each numeric feature using z-score (mean=0, std=1) — trained on train split only
6. **Export** two Excel files + plots + scaler metadata

The scaler parameters (means/stds) are saved in `code/model/scaler_params.json` so the next phases can apply the same transformation to new data.
