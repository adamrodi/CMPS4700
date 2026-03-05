# Dataset Options for CMPS 470/570 Project

Group B: Adam Rodi, Ron Logarbo

## Project Requirements Summary (from Project_4570.pdf)

- Classify object labels using ANN, SVM, DT, and K-NN
- Pipeline: Data -> Read/PreProcess -> Feature-Generator -> Classifiers + Analysis
- Analysis: Correlation and PCA
- Data split: 60/20/20 (train/validation/test) OR K-fold cross validation
- Performance metrics: Sensitivity, Specificity, Accuracy, F1, AUC
- Plots: Epoch-error curves (train + validation), distribution of splits, performance comparison across models
- Exports: Features, Model parameters, Predictions, Performance scores, Plots
- Program must read from INPUT/TRAIN and INPUT/TEST, export to OUTPUT and MODEL
- Repo structure: CODE/(INPUT, OUTPUT, MODEL), DOC/, OTHER/

---

## Option 1: Dry Bean Dataset

**Link:** https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset

### About the Data
- 13,611 samples of dry beans
- 7 classes: Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira
- 16 features (ALL numeric):
  - Dimensional (12): Area, Perimeter, Major/Minor Axis Length, Aspect Ratio, Eccentricity, Convex Area, Equivalent Diameter, Extent, Solidity, Roundness, Compactness
  - Shape (4): ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
- Target column: Class (bean type)
- Source: Computer vision measurements from high-resolution camera images
- Published paper: Koklu & Ozkan (2020), Computers and Electronics in Agriculture
- License: CC0 Public Domain

### How We Would Achieve Project Goals

**Preprocessing:**
- Check for missing values and outliers
- Normalize/standardize features (important for SVM and K-NN which are distance-sensitive)
- Split into 60/20/20 train/validation/test sets

**Feature Extraction & Analysis:**
- All 16 features are already numeric and ready to use
- Correlation matrix will show meaningful relationships (e.g., Area correlates with Perimeter, Equivalent Diameter)
- PCA will reduce 16 dimensions and show which shape/size features explain the most variance
- Feature distributions via histograms/boxplots per class

**Model Building:**
- Train ANN, SVM, DT, K-NN on the training set
- Tune parameters using validation set (e.g., K value for K-NN, kernel for SVM, tree depth for DT, hidden layers for ANN)
- Epoch-error curves for ANN training

**Model Assessment:**
- Confusion matrix for each classifier (7x7)
- Sensitivity, Specificity, Accuracy, F1, AUC per classifier
- Compare all 4 classifiers on the same performance plot

### Pros
- Real-world data with a published research paper (citable, adds credibility)
- Large sample size (13.6K) — very comfortable 60/20/20 split
- All numeric features — no encoding needed, PCA and correlation work immediately
- 7 classes creates interesting classifier differences (some beans are hard to distinguish, others easy)
- Clean dataset with no missing values
- Features have clear physical meaning — easy to interpret and explain

### Cons
- Not the most exciting topic (beans)
- The dataset is almost too clean — less preprocessing to discuss in the report
- 7 classes means 7x7 confusion matrices which can be dense to present

---

## Option 2: Stellar Classification (SDSS17)

**Link:** https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

### About the Data
- 100,000 observations from the Sloan Digital Sky Survey
- 3 classes: Star, Galaxy, Quasar
- 17 columns total, key features:
  - Photometric filters: u, g, r, i, z (ultraviolet through infrared)
  - Positional: alpha (Right Ascension), delta (Declination)
  - redshift (wavelength increase — very important predictor)
  - Identifiers to drop: obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID
- Target column: class (Star/Galaxy/Quasar)
- Source: Sloan Digital Sky Survey Data Release 17

### How We Would Achieve Project Goals

**Preprocessing:**
- Drop identifier columns (obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID) — good preprocessing story
- Subsample from 100K to ~5-10K for manageable training times
- Normalize remaining numeric features
- Split 60/20/20

**Feature Extraction & Analysis:**
- After dropping IDs, ~7-8 usable numeric features (u, g, r, i, z, alpha, delta, redshift)
- Correlation analysis between photometric bands (u, g, r, i, z are related)
- PCA on the spectral features
- Redshift is likely the dominant feature for separating quasars — interesting to show in analysis

**Model Building:**
- Train all 4 classifiers
- Redshift will likely dominate — good discussion point about feature importance
- ANN epoch-error curves

**Model Assessment:**
- 3x3 confusion matrices (cleaner to present than 7x7)
- Performance metrics across all 4 classifiers
- Compare how each handles the 3-class problem

### Pros
- Cool, interesting topic (astronomy)
- Real-world scientific data
- 100K samples — can subsample to any size needed
- Dropping ID columns gives a clear preprocessing narrative
- 3 classes keeps confusion matrices and analysis clean

### Cons
- Many columns are IDs that must be dropped — after cleaning, only ~7-8 real features for PCA
- Redshift alone may nearly perfectly classify quasars, making classifiers look trivially good
- Need to subsample (100K is overkill and slow for ANN training)
- Astronomical features may be harder to explain/interpret if you're not familiar with the domain

---

## Option 3: Smartwatch Sleep Tracking Dataset (2018-2025)

**Link:** https://www.kaggle.com/datasets/mirzayasirabdullah07/smartwatch-sleep-tracking-dataset-20182025

### About the Data
- 20,000 sleep sessions from 2,000 simulated smartwatch users
- Target: daily_label (sleep quality classification — exact classes TBD, likely Good/Fair/Poor or similar)
- Also has: sleep_score (regression target, not needed for this project)
- 45 columns including:
  - Physiological: heart rate, SpO2, respiration rate
  - Behavioral: stress level, caffeine intake, screen time
  - Environmental factors
  - Sleep metrics
- Synthetic dataset (no real wearable data)
- License: CC BY 4.0

### How We Would Achieve Project Goals

**Preprocessing:**
- Significant work needed: 45 columns require feature selection or dimensionality reduction
- Handle any categorical features (encode or drop)
- Check for and handle missing values
- Normalize numeric features
- Possibly drop redundant or low-variance features before modeling
- Split 60/20/20

**Feature Extraction & Analysis:**
- Rich feature space for PCA — 45 dimensions reduced to principal components
- Correlation matrix will be large (45x45) but will show clusters of related features (e.g., physiological signals correlating with each other)
- Can discuss feature importance and selection as part of the pipeline
- Distribution plots for key features grouped by sleep quality class

**Model Building:**
- Train all 4 classifiers
- Feature selection becomes important — can compare model performance with all features vs. PCA-reduced features
- ANN epoch-error curves
- More tuning opportunities due to high dimensionality

**Model Assessment:**
- Confusion matrix per classifier
- Performance metrics comparison
- Interesting discussion: which features matter most for sleep quality prediction?

### Pros
- Interesting, relatable topic (everyone sleeps)
- 20K samples — plenty for 60/20/20 split
- 45 features gives a rich PCA and correlation analysis story
- Lots of preprocessing to discuss in the report (feature selection, handling high dimensionality)
- Mix of feature types (physiological, behavioral, environmental) makes analysis interesting

### Cons
- Synthetic data (not real smartwatch readings) — less credible than real-world data
- 45 columns is a lot — need to invest time in understanding and selecting features before modeling
- Exact target classes (daily_label values) unknown until you download and inspect
- More work overall compared to Dry Bean which is ready to go
- No published paper behind it

---

## Quick Comparison

| Criteria                  | Dry Bean       | Stellar        | Smartwatch Sleep |
|---------------------------|----------------|----------------|------------------|
| Samples                   | 13,611         | 100,000        | 20,000           |
| Usable Numeric Features   | 16             | ~7-8           | ~45 (needs selection) |
| Classes                   | 7              | 3              | TBD (~3)         |
| Real vs Synthetic         | Real           | Real           | Synthetic        |
| Preprocessing Effort      | Low            | Medium         | High             |
| PCA/Correlation Richness  | High           | Medium         | Very High        |
| Topic Interest            | Low            | High           | High             |
| Ready to Use              | Immediately    | After cleanup  | After significant cleanup |
| Citable Paper             | Yes            | No             | No               |
