#Version: v1.0
#Date Last Updated: 03-28-2026

#%% MODULE BEGINS
module_name = "preprocess"

"""
Version: v1.0

Description:
    Preprocessing pipeline for CMPS 470 stellar classification project.

Authors:
    Group B

Date Created     : 03-28-2026
Date Last Updated: 03-28-2026

Doc:
    Reads raw data, validates schema, drops identifier columns, performs
    stratified 60/20/20 split, applies train-fit z-score scaling, exports
    Excel files and preprocessing plots, and writes summary metadata.

Notes:
    Relative paths are resolved from this file location.
"""

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import json
from copy import deepcopy as dpcpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLASS_COLUMN = "class"
DROP_COLUMNS = (
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "plate",
    "MJD",
    "fiber_ID",
)
FEATURE_COLUMNS = ("u", "g", "r", "i", "z", "alpha", "delta", "redshift")
RANDOM_SEED = 4700
SPLIT_RATIOS = (0.60, 0.20, 0.20)
SPLIT_LABELS = ("train", "validation", "test")

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BAR_COLOR_PRIMARY = "#1f77b4"
BAR_COLOR_SECONDARY = "#ff7f0e"
FIGURE_DPI = 140
HIST_CLIP_PCT = 0.005  # percentile clip applied per feature to exclude sentinel outliers


#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here



#Function definitions Start Here
def _ensure_output_dirs(paths):
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["test_dir"].mkdir(parents=True, exist_ok=True)
    paths["train_dir"].mkdir(parents=True, exist_ok=True)
    paths["validation_dir"].mkdir(parents=True, exist_ok=True)
#


def _get_paths():
    code_dir = Path(__file__).resolve().parent
    input_path = code_dir / "input" / "star_classification.csv"
    model_dir  = code_dir / "model"
    output_dir = code_dir / "output"

    return {
        "input_path":      input_path,
        "model_dir":       model_dir,
        "output_dir":      output_dir,
        "test_dir":        code_dir / "input" / "test",
        "train_dir":       code_dir / "input" / "train",
        "validation_dir":  code_dir / "input" / "validation",
    }
#


def _validate_schema(raw_df):
    required_columns = set(DROP_COLUMNS) | set(FEATURE_COLUMNS) | {CLASS_COLUMN}
    missing_columns = sorted(required_columns - set(raw_df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")
#


def _build_clean_dataframe(raw_df):
    clean_df = raw_df.drop(columns=list(DROP_COLUMNS)).copy()
    selected_columns = list(FEATURE_COLUMNS) + [CLASS_COLUMN]
    clean_df = clean_df[selected_columns]
    clean_df = clean_df.dropna().reset_index(drop=True)
    return clean_df
#


def _stratified_three_way_split(clean_df):
    rng = np.random.default_rng(RANDOM_SEED)

    train_indices = []
    validation_indices = []
    test_indices = []

    for class_value in sorted(clean_df[CLASS_COLUMN].unique()):
        class_mask = clean_df[CLASS_COLUMN] == class_value
        class_indices = clean_df.index[class_mask].to_numpy()
        shuffled_indices = rng.permutation(class_indices)

        class_count = shuffled_indices.size
        train_count = int(np.floor(class_count * SPLIT_RATIOS[0]))
        validation_count = int(np.floor(class_count * SPLIT_RATIOS[1]))
        test_count = class_count - train_count - validation_count

        train_stop = train_count
        validation_stop = train_count + validation_count
        test_stop = validation_stop + test_count

        train_indices.extend(shuffled_indices[:train_stop].tolist())
        validation_indices.extend(shuffled_indices[train_stop:validation_stop].tolist())
        test_indices.extend(shuffled_indices[validation_stop:test_stop].tolist())

    train_df = clean_df.loc[sorted(train_indices)].reset_index(drop=True)
    validation_df = clean_df.loc[sorted(validation_indices)].reset_index(drop=True)
    test_df = clean_df.loc[sorted(test_indices)].reset_index(drop=True)

    return {
        SPLIT_LABELS[0]: train_df,
        SPLIT_LABELS[1]: validation_df,
        SPLIT_LABELS[2]: test_df,
    }
#


def _zscore_scale_split(split_frames):
    scaled_frames = dpcpy(split_frames)

    train_features = scaled_frames[SPLIT_LABELS[0]].loc[:, FEATURE_COLUMNS]
    feature_means = train_features.mean(axis=0)
    feature_stds = train_features.std(axis=0, ddof=0).replace(0.0, 1.0)

    for split_label in SPLIT_LABELS:
        split_features = scaled_frames[split_label].loc[:, FEATURE_COLUMNS]
        scaled_features = (split_features - feature_means) / feature_stds
        scaled_frames[split_label].loc[:, FEATURE_COLUMNS] = scaled_features

    scaler_parameters = {
        "means": {column: float(feature_means[column]) for column in FEATURE_COLUMNS},
        "stds": {column: float(feature_stds[column]) for column in FEATURE_COLUMNS},
    }
    return scaled_frames, scaler_parameters
#


def _build_preprocessed_export_table(scaled_split_frames):
    export_frames = []
    for split_label in SPLIT_LABELS:
        split_df = scaled_split_frames[split_label].copy()
        split_df.insert(0, "split", split_label)
        export_frames.append(split_df)

    preprocessed_df = pd.concat(export_frames, axis=0, ignore_index=True)
    return preprocessed_df
#


def _plot_class_distribution(clean_df, output_path):
    class_counts = clean_df[CLASS_COLUMN].value_counts().sort_index()

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(class_counts.index, class_counts.values, color=BAR_COLOR_PRIMARY)
    axis.set_title("Class Distribution (Clean Data)")
    axis.set_xlabel("Class")
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_split_distribution(split_frames, output_path):
    class_names = sorted(split_frames[SPLIT_LABELS[0]][CLASS_COLUMN].unique())
    counts_matrix = np.zeros((len(SPLIT_LABELS), len(class_names)), dtype=int)

    for row_index, split_label in enumerate(SPLIT_LABELS):
        counts_series = split_frames[split_label][CLASS_COLUMN].value_counts()
        for col_index, class_name in enumerate(class_names):
            counts_matrix[row_index, col_index] = int(counts_series.get(class_name, 0))

    x_positions = np.arange(len(class_names))
    width = 0.25

    fig, axis = plt.subplots(figsize=(9, 5))
    for row_index, split_label in enumerate(SPLIT_LABELS):
        axis.bar(
            x_positions + (row_index - 1) * width,
            counts_matrix[row_index],
            width=width,
            label=split_label,
        )

    axis.set_title("Class Distribution by Split")
    axis.set_xlabel("Class")
    axis.set_ylabel("Count")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(class_names)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_feature_before_after(clean_df, preprocessed_df, output_path):
    # 2-row grid: top row = before z-score, bottom row = after z-score.
    # Each feature gets its own column with an independent x-axis per row,
    # and percentile-based limits to exclude sentinel outlier values (-9999).
    n_features = len(FEATURE_COLUMNS)
    figure, axes = plt.subplots(2, n_features, figsize=(22, 8))

    for col_index, feature_name in enumerate(FEATURE_COLUMNS):
        clean_vals  = clean_df[feature_name]
        scaled_vals = preprocessed_df[feature_name]

        lo_clean = float(clean_vals.quantile(HIST_CLIP_PCT))
        hi_clean = float(clean_vals.quantile(1.0 - HIST_CLIP_PCT))

        lo_scaled = float(scaled_vals.quantile(HIST_CLIP_PCT))
        hi_scaled = float(scaled_vals.quantile(1.0 - HIST_CLIP_PCT))

        top_axis = axes[0, col_index]
        top_axis.hist(
            clean_vals, bins=40, density=True,
            color=BAR_COLOR_PRIMARY, alpha=0.8,
            range=(lo_clean, hi_clean),
        )
        top_axis.set_xlim(lo_clean, hi_clean)
        top_axis.set_title(feature_name)
        top_axis.grid(axis="y", alpha=0.2)

        bot_axis = axes[1, col_index]
        bot_axis.hist(
            scaled_vals, bins=40, density=True,
            color=BAR_COLOR_SECONDARY, alpha=0.8,
            range=(lo_scaled, hi_scaled),
        )
        bot_axis.set_xlim(lo_scaled, hi_scaled)
        bot_axis.grid(axis="y", alpha=0.2)
    #

    axes[0, 0].set_ylabel("Before Z-Score\nDensity")
    axes[1, 0].set_ylabel("After Z-Score\nDensity")
    figure.suptitle("Feature Distributions: Before vs After Z-Score Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
#


def _export_excel(raw_df, preprocessed_df, paths):
    raw_output_path = paths["output_dir"] / "raw_data.xlsx"
    preprocessed_output_path = paths["output_dir"] / "preprocessed_data.xlsx"

    raw_df.to_excel(raw_output_path, index=False, sheet_name="raw_data")
    preprocessed_df.to_excel(preprocessed_output_path, index=False, sheet_name="preprocessed_data")
#


def _export_metadata(clean_df, split_frames, scaler_parameters, paths):
    split_sizes = {label: int(split_frames[label].shape[0]) for label in SPLIT_LABELS}

    split_class_distribution = {}
    for split_label in SPLIT_LABELS:
        split_class_counts = split_frames[split_label][CLASS_COLUMN].value_counts().sort_index()
        split_class_distribution[split_label] = {
            class_name: int(split_class_counts[class_name])
            for class_name in split_class_counts.index
        }

    preprocessing_summary = {
        "class_column": CLASS_COLUMN,
        "drop_columns": list(DROP_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "random_seed": RANDOM_SEED,
        "split_ratios": list(SPLIT_RATIOS),
        "split_sizes": split_sizes,
        "split_class_distribution": split_class_distribution,
        "clean_row_count": int(clean_df.shape[0]),
        "scaler": "zscore_train_fit",
        "scaler_parameters": scaler_parameters,
    }

    summary_output_path = paths["output_dir"] / "preprocessing_summary.json"
    scaler_output_path = paths["model_dir"] / "scaler_params.json"

    with open(summary_output_path, "w", encoding="utf-8") as summary_file:
        json.dump(preprocessing_summary, summary_file, indent=2)

    with open(scaler_output_path, "w", encoding="utf-8") as scaler_file:
        json.dump(scaler_parameters, scaler_file, indent=2)
#


def _export_plots(clean_df, split_frames, preprocessed_df, paths):
    class_distribution_plot_path = paths["output_dir"] / "plot_class_distribution_clean.png"
    split_distribution_plot_path = paths["output_dir"] / "plot_class_distribution_by_split.png"
    feature_distribution_plot_path = paths["output_dir"] / "plot_features_before_after_zscore.png"

    _plot_class_distribution(clean_df, class_distribution_plot_path)
    _plot_split_distribution(split_frames, split_distribution_plot_path)
    _plot_feature_before_after(clean_df, preprocessed_df, feature_distribution_plot_path)
#


def _export_splits_csv(scaled_split_frames, paths):
    # Write each split as a standalone CSV into its input subdirectory
    dir_keys = {
        SPLIT_LABELS[0]: paths["train_dir"],
        SPLIT_LABELS[1]: paths["validation_dir"],
        SPLIT_LABELS[2]: paths["test_dir"],
    }
    for split_label in SPLIT_LABELS:
        output_path = dir_keys[split_label] / f"{split_label}.csv"
        scaled_split_frames[split_label].to_csv(output_path, index=False)
    #
#


def main():
    paths = _get_paths()
    _ensure_output_dirs(paths)

    raw_df = pd.read_csv(paths["input_path"])
    _validate_schema(raw_df)

    clean_df = _build_clean_dataframe(raw_df)
    split_frames = _stratified_three_way_split(clean_df)
    scaled_split_frames, scaler_parameters = _zscore_scale_split(split_frames)

    preprocessed_df = _build_preprocessed_export_table(scaled_split_frames)

    _export_splits_csv(scaled_split_frames, paths)
    _export_excel(raw_df, preprocessed_df, paths)
    _export_metadata(clean_df, split_frames, scaler_parameters, paths)
    _export_plots(clean_df, split_frames, preprocessed_df, paths)

    print("PA1 preprocessing completed successfully.")
    print(f"Train CSV    : {(paths['train_dir']      / 'train.csv').as_posix()}")
    print(f"Validation CSV: {(paths['validation_dir'] / 'validation.csv').as_posix()}")
    print(f"Test CSV     : {(paths['test_dir']        / 'test.csv').as_posix()}")
    print(f"Preprocessed Excel: {(paths['output_dir'] / 'preprocessed_data.xlsx').as_posix()}")
#


#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":

    print(f'"{module_name}" module begins.')

    #TEST Code
    main()