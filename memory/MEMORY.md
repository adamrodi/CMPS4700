# Project Memory

## Dataset
- **Chosen: Stellar Classification (SDSS DR17)**
- Kaggle: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
- 100,000 observations; subsample to ~10,000 for training
- 3 classes: STAR, GALAXY, QSO
- 18 raw columns; drop 9 identifiers: obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID
- 8 usable features: u, g, r, i, z (photometric), alpha, delta (positional), redshift
- redshift is the dominant feature for quasar separation

## Repo Structure (post-cleanup)
```
C0/              — Challenge 0 (completed)
Notebooks/       — adam.ipynb, ron.ipynb (working notebooks)
code/
  input/train/   — place star_classification.csv here (not committed)
  input/test/    — test split goes here
  output/        — plots, predictions, scores (not committed)
  model/         — saved model params (not committed)
  module_tmp.py  — module template (reference)
doc/             — Project_4570.pdf, pptx files, GroupB_proposal.pptx
other/           — dataset_options.md, make_proposal.py
```

## Project Requirements
- Classifiers: ANN, SVM, DT, K-NN (scikit-learn)
- Pipeline: Data -> Read/PreProcess -> Feature-Generator -> Classifiers + Analysis
- Split: 60/20/20 (train/validation/test)
- Metrics: Sensitivity, Specificity, Accuracy, F1, AUC
- Analysis: Correlation matrix, PCA
- Module template: code/module_tmp.py

## Coding Conventions
- Vectorized numpy/pandas (no loops)
- Self-explanatory variable names, generic coding (no hard-coded constants)
- deepcopy imported as dpcpy
- Alphabetically sorted declarations
- Relative paths
- No if-blocks or declarations inside loops
- Initialize arrays when size is known
- `#` at end of each code block
