#Version: v1.0
#Date Last Updated: 04-12-2026

#%% MODULE BEGINS
module_name = "feature_extract"

"""
Version: v1.0

Description:
    Feature extraction pipeline for CMPS 470 stellar classification project.
    Computes Pearson correlation matrix on training features, fits PCA on the
    training split, applies the transform to all splits, and exports extracted
    features, model parameters, and visualizations.

Authors:
    Group B

Date Created     : 04-12-2026
Date Last Updated: 04-12-2026

Doc:
    Input : code/output/preprocessed_data.xlsx  (PA1 output, z-score scaled)
    Output: code/output/features_data.xlsx
            code/model/pca_params.json
            code/output/plot_correlation_heatmap.png
            code/output/plot_pca_cumulative_variance.png
            code/output/plot_pca_scatter_pc1_pc2.png
            code/output/plot_pca_scree.png

Notes:
    Relative paths are resolved from this file location.
    PCA is fit on the training split only to prevent data leakage.
"""

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import json
from copy import deepcopy as dpcpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLASS_COLUMN       = "class"
FEATURE_COLUMNS    = ("u", "g", "r", "i", "z", "alpha", "delta", "redshift")
SPLIT_LABELS       = ("train", "validation", "test")
VARIANCE_THRESHOLD = 0.95

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BAR_COLOR_PRIMARY = "#1f77b4"
CLASS_COLORS      = {"GALAXY": "#1f77b4", "QSO": "#ff7f0e", "STAR": "#2ca02c"}
CLASS_MARKERS     = {"GALAXY": "o",       "QSO": "s",       "STAR": "^"}
CUTOFF_COLOR      = "orange"
FIGURE_DPI        = 140
HEATMAP_CMAP      = "coolwarm"
SCATTER_ALPHA     = 0.15
SCATTER_IQR_MULT  = 3.0    # axis limits: Q1 - mult*IQR to Q3 + mult*IQR per PC
SCATTER_PC_PAIRS  = ((1, 2), (1, 3), (2, 3), (3, 4), (4, 5))
SCATTER_SIZE      = 8
THRESHOLD_COLOR   = "red"

#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here


#Class definitions Start Here


#Function definitions Start Here
def _apply_pca(split_frames, pca, n_components):
    # Transform all splits using the already-fitted PCA object
    pc_columns    = [f"PC{k}" for k in range(1, n_components + 1)]
    pca_frames    = {label: None for label in SPLIT_LABELS}

    for split_label in SPLIT_LABELS:
        feature_values = split_frames[split_label][list(FEATURE_COLUMNS)].values
        transformed    = pca.transform(feature_values)[:, :n_components]

        split_df                = pd.DataFrame(transformed, columns=pc_columns)
        split_df.insert(0, "split", split_label)
        split_df[CLASS_COLUMN]  = split_frames[split_label][CLASS_COLUMN].values

        pca_frames[split_label] = dpcpy(split_df)
    #

    return pca_frames
#


def _build_features_export_table(pca_frames):
    # Concatenate all splits in train / validation / test order
    export_frames = []
    for split_label in SPLIT_LABELS:
        export_frames.append(pca_frames[split_label].copy())
    #

    return pd.concat(export_frames, axis=0, ignore_index=True)
#


def _compute_correlation_matrix(train_df):
    # Pearson correlation matrix computed on training features only
    feature_values = train_df[list(FEATURE_COLUMNS)].values
    return np.corrcoef(feature_values.T)
#


def _ensure_output_dirs(paths):
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
#


def _export_excel(features_df, corr_matrix, paths):
    # Write features_data.xlsx with two sheets
    output_path = paths["output_dir"] / "features_data.xlsx"
    corr_df     = pd.DataFrame(
        corr_matrix,
        index=list(FEATURE_COLUMNS),
        columns=list(FEATURE_COLUMNS),
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        features_df.to_excel(writer, index=False, sheet_name="features_data")
        corr_df.to_excel(writer, sheet_name="correlation_matrix")
#


def _export_plots(pca, n_components, pca_frames, corr_matrix, paths):
    # Resolve output paths and dispatch to each plot function
    corr_heatmap_path = paths["output_dir"] / "plot_correlation_heatmap.png"
    cumvar_path       = paths["output_dir"] / "plot_pca_cumulative_variance.png"
    scatter_path      = paths["output_dir"] / "plot_pca_scatter_pairs.png"
    scree_path        = paths["output_dir"] / "plot_pca_scree.png"

    _plot_correlation_heatmap(corr_matrix, corr_heatmap_path)
    _plot_cumulative_variance(pca, n_components, cumvar_path)
    _plot_pca_scatter(pca_frames, n_components, scatter_path)
    _plot_scree(pca, n_components, scree_path)
#


def _export_pca_params(pca, n_components, paths):
    # Write PCA model parameters to JSON for use by future model modules
    pc_keys = [f"PC{k}" for k in range(1, n_components + 1)]

    explained_variance_ratios = [float(v) for v in pca.explained_variance_ratio_]
    cumulative_variance       = [float(v) for v in np.cumsum(pca.explained_variance_ratio_)]

    loadings = {
        pc_keys[k]: {
            feature: float(pca.components_[k, j])
            for j, feature in enumerate(FEATURE_COLUMNS)
        }
        for k in range(n_components)
    }

    pca_params = {
        "cumulative_variance":      cumulative_variance,
        "explained_variance_ratios": explained_variance_ratios,
        "feature_columns":          list(FEATURE_COLUMNS),
        "loadings":                 loadings,
        "n_components":             n_components,
        "variance_threshold":       float(VARIANCE_THRESHOLD),
    }

    pca_params_path = paths["model_dir"] / "pca_params.json"
    with open(pca_params_path, "w", encoding="utf-8") as params_file:
        json.dump(pca_params, params_file, indent=2)
#


def _fit_pca(train_df):
    # Fit full PCA on training features; select n_components to meet variance threshold
    feature_values    = train_df[list(FEATURE_COLUMNS)].values
    pca               = PCA(svd_solver="full")
    pca.fit(feature_values)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components        = int(np.argmax(cumulative_variance >= VARIANCE_THRESHOLD)) + 1

    return pca, n_components
#


def _get_paths():
    code_dir   = Path(__file__).resolve().parent
    model_dir  = code_dir / "model"
    output_dir = code_dir / "output"

    return {
        "model_dir":       model_dir,
        "output_dir":      output_dir,
        "test_path":       code_dir / "input" / "test"       / "test.csv",
        "train_path":      code_dir / "input" / "train"      / "train.csv",
        "validation_path": code_dir / "input" / "validation" / "validation.csv",
    }
#


def _iqr_limits(series, mult):
    # Compute outlier-resistant axis limits: Q1 - mult*IQR to Q3 + mult*IQR
    q1  = float(series.quantile(0.25))
    q3  = float(series.quantile(0.75))
    iqr = q3 - q1
    return (q1 - mult * iqr, q3 + mult * iqr)
#


def _load_splits(paths):
    # Read each split from its dedicated CSV in code/input/<split>/
    path_keys    = {
        SPLIT_LABELS[0]: paths["train_path"],
        SPLIT_LABELS[1]: paths["validation_path"],
        SPLIT_LABELS[2]: paths["test_path"],
    }
    split_frames = {label: None for label in SPLIT_LABELS}

    for split_label in SPLIT_LABELS:
        split_frames[split_label] = dpcpy(pd.read_csv(path_keys[split_label]))
    #

    return split_frames
#


def _plot_correlation_heatmap(corr_matrix, output_path):
    # Heatmap of the 8x8 Pearson correlation matrix with cell annotations
    n_features      = len(FEATURE_COLUMNS)
    tick_positions  = list(range(n_features))

    fig, axis = plt.subplots(figsize=(9, 7))
    image     = axis.imshow(corr_matrix, cmap=HEATMAP_CMAP, vmin=-1, vmax=1)
    fig.colorbar(image, ax=axis, label="Pearson r")

    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
    axis.set_yticklabels(FEATURE_COLUMNS)
    axis.set_title("Training Set Feature Correlation Matrix")

    for row_index in range(n_features):
        for col_index in range(n_features):
            axis.text(
                col_index, row_index,
                f"{corr_matrix[row_index, col_index]:.2f}",
                ha="center", va="center", fontsize=8,
            )
        #
    #

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_cumulative_variance(pca, n_components, output_path):
    # Line plot of cumulative explained variance with threshold and cutoff markers
    x_positions         = list(range(1, len(FEATURE_COLUMNS) + 1))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_).tolist()

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        x_positions, cumulative_variance,
        marker="o", color=BAR_COLOR_PRIMARY, linestyle="-",
        label="Cumulative variance",
    )
    axis.axhline(
        VARIANCE_THRESHOLD, color=THRESHOLD_COLOR, linestyle="--",
        label=f"{int(VARIANCE_THRESHOLD * 100)}% threshold",
    )
    axis.axvline(
        n_components, color=CUTOFF_COLOR, linestyle=":",
        label=f"n={n_components}",
    )

    axis.set_title("Cumulative Explained Variance by PCA Components")
    axis.set_xlabel("Number of Components")
    axis.set_ylabel("Cumulative Explained Variance")
    axis.set_xticks(x_positions)
    axis.set_ylim(0.0, 1.05)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_pca_scatter(pca_frames, n_components, output_path):
    # Grid of scatter plots for selected PC pairs (training set only).
    # Axes use IQR-based limits to show structure while excluding extreme outliers.
    train_df    = pca_frames["train"]
    valid_pairs = tuple((a, b) for (a, b) in SCATTER_PC_PAIRS if a <= n_components and b <= n_components)
    n_pairs     = len(valid_pairs)
    n_cols      = 3
    n_rows      = (n_pairs + n_cols - 1) // n_cols

    class_masks = {
        cls: train_df[train_df[CLASS_COLUMN] == cls]
        for cls in sorted(CLASS_COLORS.keys())
    }
    pc_limits   = {
        k: _iqr_limits(train_df[f"PC{k}"], SCATTER_IQR_MULT)
        for k in range(1, n_components + 1)
    }

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes_flat    = axes.ravel()

    for pair_index, (pc_a, pc_b) in enumerate(valid_pairs):
        axis  = axes_flat[pair_index]
        col_a = f"PC{pc_a}"
        col_b = f"PC{pc_b}"

        for cls in sorted(CLASS_COLORS.keys()):
            mask = class_masks[cls]
            axis.scatter(
                mask[col_a], mask[col_b],
                c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                s=SCATTER_SIZE, alpha=SCATTER_ALPHA, label=cls,
            )
        #

        axis.set_xlim(pc_limits[pc_a])
        axis.set_ylim(pc_limits[pc_b])
        axis.set_xlabel(col_a)
        axis.set_ylabel(col_b)
        axis.grid(alpha=0.2)
    #

    for unused_index in range(n_pairs, n_rows * n_cols):
        axes_flat[unused_index].set_visible(False)
    #

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(CLASS_COLORS),
        markerscale=3, fontsize=10, bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle("PCA Scatter Plots by Component Pairs (Training Set)", y=1.05)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
#


def _plot_scree(pca, n_components, output_path):
    # Bar chart of individual explained variance per component with cutoff line
    x_positions = list(range(1, len(FEATURE_COLUMNS) + 1))

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        x_positions, pca.explained_variance_ratio_,
        color=BAR_COLOR_PRIMARY, label="Individual variance",
    )
    axis.axvline(
        n_components + 0.5, color=THRESHOLD_COLOR, linestyle="--",
        label=f"{int(VARIANCE_THRESHOLD * 100)}% threshold at n={n_components}",
    )

    axis.set_title("PCA Scree Plot")
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Explained Variance Ratio")
    axis.set_xticks(x_positions)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def main():
    paths = _get_paths()
    _ensure_output_dirs(paths)

    split_frames          = _load_splits(paths)
    corr_matrix           = _compute_correlation_matrix(split_frames["train"])
    pca, n_components     = _fit_pca(split_frames["train"])
    pca_frames            = _apply_pca(split_frames, pca, n_components)
    features_df           = _build_features_export_table(pca_frames)

    _export_pca_params(pca, n_components, paths)
    _export_excel(features_df, corr_matrix, paths)
    _export_plots(pca, n_components, pca_frames, corr_matrix, paths)

    print("PA2 feature extraction completed successfully.")
    print(f"Features Excel : {(paths['output_dir'] / 'features_data.xlsx').as_posix()}")
    print(f"PCA params     : {(paths['model_dir']  / 'pca_params.json').as_posix()}")
    print(f"n_components   : {n_components}  ({VARIANCE_THRESHOLD:.0%} variance threshold)")
#


#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":

    print(f'"{module_name}" module begins.')

    #TEST Code
    main()
