# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course

CMPS 470/570 — Machine Learning classification project. Team: Adam Rodi, Ron Logarbo (Group B).

## Repository Structure

- `C0/` — Challenge 0 (completed): data generation, rotation/shifting transforms, visualization, simple median-based classifier with confusion matrix and performance metrics
- `Notebooks/` — Individual working notebooks (`adam.ipynb`, `ron.ipynb`) used to develop challenge solutions before merging into final submissions
- `Project/` — Semester project: ML classification app using k-NN, ANN, SVM, and Decision Tree classifiers
  - `Data.xlsx` — project dataset
  - `module_tmp.py` — Python module template with section markers (`#%% IMPORTS`, `#%% CONSTANTS`, etc.)
  - `Project_4570.pdf` — project specification

## Project Architecture (from spec)

Pipeline: Data → Read/PreProcess → Feature-Generator → Classifier + Analysis

- **Classifiers**: ANN, SVM, DT, K-NN
- **Analysis**: Correlation, PCA
- **Exports**: Features, Model, Performance-epoch curves, Predictions, Performance scores, Plots

Modules should follow the template in `module_tmp.py` with sections: IMPORTS, USER INTERFACE, CONSTANTS, CONFIGURATION, INITIALIZATIONS, DECLARATIONS, MAIN CODE, SELF-RUN.

## Coding Conventions

From challenge instructions and the module template:

- Use vectorized numpy/pandas operations instead of loops
- Use self-explanatory variable names
- Use generic coding — avoid hard-coded constant values
- Use `deepcopy` (imported as `dpcpy`) when copying mutable objects
- Sort declarations alphabetically
- Use relative paths
- Avoid if-blocks and declarations inside loops
- Initialize arrays when size is known

## Key Libraries

numpy, pandas, matplotlib (with `MultipleLocator` for axis ticks), scikit-learn (for project classifiers)

## Running Notebooks

```terminal
jupyter notebook
```

Notebooks should be run top-to-bottom and saved with outputs before submission.
