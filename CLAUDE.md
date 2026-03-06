# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course

CMPS 470/570 — Machine Learning classification project. Team: Adam Rodi, Ron Logarbo (Group B).

## Repository Structure

```
C0/                    — Challenge 0 (completed)
Notebooks/             — Individual working notebooks (adam.ipynb, ron.ipynb)
code/
  input/
    train/             — Training data (download SDSS17 CSV — not committed)
    test/              — Test data (not committed)
  output/              — Generated results, plots, predictions (not committed)
  model/               — Saved model parameters (not committed)
  module_tmp.py        — Python module template (reference)
doc/
  Project_4570.pdf     — Project specification
  Project_4570.pptx    — Project report template
  GroupB_proposal.pptx — Submitted proposal
other/
  dataset_options.md   — Dataset comparison notes (planning)
  make_proposal.py     — Script used to generate proposal PPTX
```

## Project

**Dataset:** Stellar Classification — Sloan Digital Sky Survey Data Release 17 (SDSS DR17)
- Source: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
- 100,000 observations; subsample to ~10,000 for manageable training times
- 3 classes: STAR, GALAXY, QSO (Quasar)
- 18 raw columns; drop 9 identifiers: obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID
- 8 usable features: u, g, r, i, z (photometric filters), alpha, delta (positional), redshift

**Pipeline:** Data → Read/PreProcess → Feature-Generator → Classifier + Analysis

**Classifiers:** ANN, SVM, DT, K-NN (scikit-learn)

**Analysis:** Correlation matrix, PCA

**Exports:** Features, Model parameters, Performance-epoch curves, Predictions, Performance scores, Plots

**Data split:** 60% train / 20% validation / 20% test

**Performance metrics:** Sensitivity, Specificity, Accuracy, F1, AUC

## Coding Conventions

From `code/module_tmp.py` and course instructions:

- Use vectorized numpy/pandas operations instead of loops
- Use self-explanatory variable names
- Use generic coding — avoid hard-coded constant values
- Use `deepcopy` (imported as `dpcpy`) when copying mutable objects
- Sort declarations alphabetically
- Use relative paths
- Avoid if-blocks and declarations inside loops
- Initialize arrays when size is known
- Add `#` comment at end of each block
- Divide code into sections: IMPORTS, USER INTERFACE, CONSTANTS, CONFIGURATION, INITIALIZATIONS, DECLARATIONS, MAIN CODE, SELF-RUN

## Key Libraries

numpy, pandas, matplotlib (with `MultipleLocator` for axis ticks), scikit-learn

## Running Notebooks

```terminal
jupyter notebook
```

Notebooks should be run top-to-bottom and saved with outputs before submission.
