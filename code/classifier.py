#Version: v1.1
#Date Last Updated: 05-01-2026

#%% MODULE BEGINS
module_name = "classifier"

"""
Version: v1.1

Description:
    Classifier stage for the CMPS 4700 stellar classification project.
    Reads PCA feature data, trains ANN, SVM, Decision Tree, and K-NN models,
    evaluates model performance, exports trained models, predictions,
    performance scores, and plots.

Authors:
    Group B

Date Created     : 04-29-2026
Date Last Updated: 05-01-2026

Doc:
    Input : code/output/features_data.xlsx
    Output: code/output/model_performance.xlsx
            code/output/model_predictions.xlsx
            code/output/plot_model_accuracy.png
            code/output/plot_model_f1.png
            code/output/plot_ann_loss_curve.png
            code/output/plot_model_metrics_combined.png
            code/output/plot_confusion_matrix_<model>.png
            code/model/classifier_params.json
            code/model/<model>_model.joblib

Notes:
    Relative paths are resolved from this file location.
    Models are trained on the training split and evaluated on all splits
    (train, validation, test).
"""

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
#

import json
from copy import deepcopy as dpcpy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLASS_COLUMN    = "class"
FEATURE_COLUMNS = ("PC1", "PC2", "PC3", "PC4", "PC5")
SPLIT_COLUMN    = "split"
SPLIT_LABELS    = ("train", "validation", "test")

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FIGURE_DPI      = 140
RANDOM_SEED     = 4700

#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here


#Class definitions Start Here


#Function definitions Start Here
def _build_models():
    models = {
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
        ),
        "K-NN": KNeighborsClassifier(
            n_neighbors=7,
        ),
        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                dual=False,
                max_iter=5000,
                random_state=RANDOM_SEED,
            ),
            cv=3,
        ),
    }

    return models
#


def _compute_specificity(y_true, y_pred, class_labels):
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    specificities = []

    # Compute per-class TN and FP to derive class-level specificity
    for class_index in range(len(class_labels)):
        true_negative = (
            matrix.sum()
            - matrix[class_index, :].sum()
            - matrix[:, class_index].sum()
            + matrix[class_index, class_index]
        )
        false_positive = matrix[:, class_index].sum() - matrix[class_index, class_index]
        denominator = true_negative + false_positive

        if denominator == 0:
            specificity = 0.0
        else:
            specificity = true_negative / denominator
        #

        specificities.append(specificity)
    #

    return float(np.mean(specificities))
#


def _ensure_output_dirs(paths):
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
#


def _evaluate_model(model, split_label, x_values, y_values, label_encoder):
    class_labels = np.arange(len(label_encoder.classes_))

    y_pred = model.predict(x_values)

    performance_row = {
        "split": split_label,
        "accuracy": float(accuracy_score(y_values, y_pred)),
        "auc_ovr_weighted": None,
        "f1_weighted": float(f1_score(y_values, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_values, y_pred, average="weighted", zero_division=0)),
        "sensitivity_recall_weighted": float(recall_score(y_values, y_pred, average="weighted", zero_division=0)),
        "specificity_macro": float(_compute_specificity(y_values, y_pred, class_labels)),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_values)
        performance_row["auc_ovr_weighted"] = float(
            roc_auc_score(y_values, y_prob, multi_class="ovr", average="weighted")
        )
    #

    predictions_df = pd.DataFrame(
        {
            "split": split_label,
            "actual": label_encoder.inverse_transform(y_values),
            "predicted": label_encoder.inverse_transform(y_pred),
        }
    )

    return performance_row, predictions_df
#


def _export_classifier_params(results_df, paths):
    best_test_row = (
        results_df[results_df["split"] == "test"]
        .sort_values(by="f1_weighted", ascending=False)
        .iloc[0]
    )

    classifier_params = {
        "best_model_by_test_f1": str(best_test_row["model"]),
        "class_column": CLASS_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "models": sorted(results_df["model"].unique().tolist()),
        "random_seed": RANDOM_SEED,
        "selection_metric": "test f1_weighted",
        "split_column": SPLIT_COLUMN,
        "splits_evaluated": list(SPLIT_LABELS),
    }

    params_path = paths["model_dir"] / "classifier_params.json"
    with open(params_path, "w", encoding="utf-8") as params_file:
        json.dump(classifier_params, params_file, indent=2)
    #
#


def _export_excel(results_df, predictions_df, paths):
    performance_path = paths["output_dir"] / "model_performance.xlsx"
    predictions_path = paths["output_dir"] / "model_predictions.xlsx"

    results_df.to_excel(performance_path, index=False, sheet_name="performance")
    predictions_df.to_excel(predictions_path, index=False, sheet_name="predictions")
#


def _export_models(trained_models, paths):
    safe_model_names = {
        "ANN": "ann",
        "Decision Tree": "decision_tree",
        "K-NN": "knn",
        "SVM": "svm",
    }

    # Serialize each trained model to a .joblib file in the model directory
    for model_name, model in trained_models.items():
        model_path = paths["model_dir"] / f"{safe_model_names[model_name]}_model.joblib"
        joblib.dump(model, model_path)
    #
#


def _export_plots(results_df, trained_models, split_frames, label_encoder, paths):
    _plot_metric_comparison(results_df, "accuracy", paths["output_dir"] / "plot_model_accuracy.png")
    _plot_metric_comparison(results_df, "f1_weighted", paths["output_dir"] / "plot_model_f1.png")

    # Grouped bar chart of all metrics on one figure
    _plot_metrics_grouped(results_df, paths["output_dir"] / "plot_model_metrics_combined.png")

    # ANN epoch-error (training loss) curve
    _plot_ann_loss_curve(trained_models, paths)

    test_df = split_frames["test"]
    x_test  = test_df[list(FEATURE_COLUMNS)].values
    y_test  = test_df["encoded_class"].values

    # Confusion matrix for each trained model on the test split
    for model_name, model in trained_models.items():
        output_name = model_name.lower().replace(" ", "_").replace("-", "")
        output_path = paths["output_dir"] / f"plot_confusion_matrix_{output_name}.png"
        _plot_confusion_matrix(model_name, model, x_test, y_test, label_encoder, output_path)
    #
#


def _get_paths():
    code_dir = Path(__file__).resolve().parent
    model_dir = code_dir / "model"
    output_dir = code_dir / "output"

    return {
        "features_path": output_dir / "features_data.xlsx",
        "model_dir": model_dir,
        "output_dir": output_dir,
    }
#


def _load_feature_data(paths):
    features_df = pd.read_excel(paths["features_path"])
    required_columns = set(FEATURE_COLUMNS) | {CLASS_COLUMN, SPLIT_COLUMN}

    missing_columns = sorted(required_columns - set(features_df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")
    #

    return features_df
#


def _plot_confusion_matrix(model_name, model, x_test, y_test, label_encoder, output_path):
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)

    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)

    axis.set_title(f"Confusion Matrix: {model_name}")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")
    axis.set_xticks(np.arange(len(label_encoder.classes_)))
    axis.set_yticks(np.arange(len(label_encoder.classes_)))
    axis.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
    axis.set_yticklabels(label_encoder.classes_)

    # Annotate each cell with its integer count
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                fontsize=9,
            )
        #
    #

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_metric_comparison(results_df, metric_name, output_path):
    test_results = results_df[results_df["split"] == "test"].copy()
    test_results = test_results.sort_values(by=metric_name, ascending=False)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(test_results["model"], test_results[metric_name])
    axis.set_title(f"Test {metric_name.replace('_', ' ').title()} by Model")
    axis.set_xlabel("Model")
    axis.set_ylabel(metric_name.replace("_", " ").title())
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_ann_loss_curve(trained_models, paths):
    # Plot training loss per iteration from the fitted ANN (loss_curve_ attribute)
    ann_model   = trained_models["ANN"]
    loss_values = ann_model.loss_curve_
    iterations  = list(range(1, len(loss_values) + 1))

    fig, axis = plt.subplots(figsize=(8, 5))

    # Draw training loss curve
    axis.plot(iterations, loss_values, label="Training Loss", color="#1f77b4")

    axis.set_title("ANN Training Loss Curve (Epoch-Error)")
    axis.set_xlabel("Epoch (Iteration)")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.2)

    output_path = paths["output_dir"] / "plot_ann_loss_curve.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _plot_metrics_grouped(results_df, output_path):
    # Grouped bar chart comparing all key metrics across models for the test split
    METRIC_COLUMNS = (
        "accuracy",
        "precision_weighted",
        "sensitivity_recall_weighted",
        "specificity_macro",
        "f1_weighted",
        "auc_ovr_weighted",
    )
    METRIC_LABELS = ("Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC")

    test_results = results_df[results_df["split"] == "test"].copy()
    test_results = test_results.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
    test_results[list(METRIC_COLUMNS)] = test_results[list(METRIC_COLUMNS)].fillna(0.0)

    model_names = test_results["model"].tolist()
    n_models    = len(model_names)
    n_metrics   = len(METRIC_COLUMNS)

    x_positions = np.arange(n_models)
    bar_width   = 0.12
    offsets     = np.linspace(
        -(n_metrics - 1) / 2 * bar_width,
         (n_metrics - 1) / 2 * bar_width,
        n_metrics,
    )

    fig, axis = plt.subplots(figsize=(12, 6))

    # Draw one bar group per metric across all models
    for metric_index, (metric_col, metric_label) in enumerate(zip(METRIC_COLUMNS, METRIC_LABELS)):
        axis.bar(
            x_positions + offsets[metric_index],
            test_results[metric_col].values,
            width=bar_width,
            label=metric_label,
        )
    #

    axis.set_title("Model Performance Comparison — All Metrics (Test Split)")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(model_names)
    axis.set_ylim(0.0, 1.10)
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
#


def _split_feature_data(features_df, label_encoder):
    prepared_df = dpcpy(features_df)
    prepared_df["encoded_class"] = label_encoder.fit_transform(prepared_df[CLASS_COLUMN])

    split_frames = {
        split_label: prepared_df[prepared_df[SPLIT_COLUMN] == split_label].reset_index(drop=True)
        for split_label in SPLIT_LABELS
    }

    for split_label, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"No rows found for split: {split_label}")
        #
    #

    return split_frames
#


def _train_and_evaluate(split_frames, label_encoder):
    models = _build_models()
    trained_models = {}
    performance_rows = []
    prediction_frames = []

    train_df = split_frames["train"]
    x_train = train_df[list(FEATURE_COLUMNS)].values
    y_train = train_df["encoded_class"].values

    # Fit each model on the training split and evaluate across all splits
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(x_train, y_train)
        trained_models[model_name] = dpcpy(model)

        # Score the fitted model on every split (train, validation, test)
        for split_label in SPLIT_LABELS:
            split_df = split_frames[split_label]
            x_values = split_df[list(FEATURE_COLUMNS)].values
            y_values = split_df["encoded_class"].values

            performance_row, predictions_df = _evaluate_model(
                model,
                split_label,
                x_values,
                y_values,
                label_encoder,
            )

            performance_row["model"] = model_name
            predictions_df.insert(0, "model", model_name)

            performance_rows.append(performance_row)
            prediction_frames.append(predictions_df)
        #
    #

    results_df = pd.DataFrame(performance_rows)
    predictions_df = pd.concat(prediction_frames, axis=0, ignore_index=True)

    ordered_columns = [
        "model",
        "split",
        "accuracy",
        "precision_weighted",
        "sensitivity_recall_weighted",
        "specificity_macro",
        "f1_weighted",
        "auc_ovr_weighted",
    ]

    results_df = results_df[ordered_columns]

    return trained_models, results_df, predictions_df
#


def main():
    paths = _get_paths()
    _ensure_output_dirs(paths)

    features_df = _load_feature_data(paths)

    label_encoder = LabelEncoder()
    split_frames = _split_feature_data(features_df, label_encoder)

    trained_models, results_df, predictions_df = _train_and_evaluate(split_frames, label_encoder)

    _export_models(trained_models, paths)
    _export_excel(results_df, predictions_df, paths)
    _export_classifier_params(results_df, paths)
    _export_plots(results_df, trained_models, split_frames, label_encoder, paths)

    print("\nClassifier stage completed successfully.")
    print(f"Performance Excel: {(paths['output_dir'] / 'model_performance.xlsx').as_posix()}")
    print(f"Predictions Excel : {(paths['output_dir'] / 'model_predictions.xlsx').as_posix()}")
    print(f"Classifier params : {(paths['model_dir'] / 'classifier_params.json').as_posix()}")
    print(f"ANN loss curve    : {(paths['output_dir'] / 'plot_ann_loss_curve.png').as_posix()}")
    print(f"Combined metrics  : {(paths['output_dir'] / 'plot_model_metrics_combined.png').as_posix()}")
    print("\nModel performance:")
    print(results_df.to_string(index=False))
#


#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":

    print(f'"{module_name}" module begins.')

    #TEST Code
    main()