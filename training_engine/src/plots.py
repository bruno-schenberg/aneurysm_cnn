"""
plots.py

Plotting, metric calculation, and artifact-saving utilities for the training
engine. Called by the orchestrator at the end of each fold to produce a
complete record of training behaviour and evaluation results.
"""

import csv
import itertools
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_curve,
)


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    filename: str,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Render a confusion matrix and save it as a PNG file.

    Args:
        cm: Confusion matrix array of shape ``(n_classes, n_classes)``.
        classes: Class label strings used for axis tick labels.
        filename: Path where the PNG will be saved.
        normalize: If ``True``, row-normalise so each cell shows the fraction
            of true-class samples predicted in that column.
        title: Title string displayed above the matrix.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc_curve(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    classes: List[str],
    filename: str,
) -> None:
    """
    Re-run inference to collect class probabilities, then plot and save a ROC curve.

    Args:
        model: Trained model.
        dataloader: DataLoader over the evaluation split.
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
        classes: Class label strings.
        filename: Path where the PNG will be saved.
    """
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["image"], batch["label"]
            inputs = inputs.to(device)
            outputs = model(inputs)

            if outputs.shape[1] == 2:
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probabilities = outputs.squeeze(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    if len(np.unique(all_labels)) < 2:
        print(
            "Warning: Only one class found in evaluation labels. "
            "Cannot generate ROC curve."
        )
        return

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


# ── Metric Calculation ────────────────────────────────────────────────────────


def calculate_classification_metrics(
    detailed_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F2-Score from a list of prediction records.

    Args:
        detailed_results: List of prediction dicts.

    Returns:
        Dict with keys ``'Precision'``, ``'Recall'``, ``'F2-Score'``, each
        a float in [0, 1].
    """
    if not detailed_results:
        return {"Precision": float("nan"), "Recall": float("nan"), "F2-Score": float("nan")}

    try:
        df = pd.DataFrame(detailed_results)
        y_true = df["true_label"].astype(int)
        y_pred = df["pred_label"].astype(int)

        if len(y_true.unique()) < 2:
            print("Warning: Only one class present in true labels. Metrics will be 0.")
            return {"Precision": 0.0, "Recall": 0.0, "F2-Score": 0.0}

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f2_score = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)

        return {"Precision": precision, "Recall": recall, "F2-Score": f2_score}

    except Exception as e:
        print(f"An error occurred during metric calculation: {e}")
        return {"Precision": float("nan"), "Recall": float("nan"), "F2-Score": float("nan")}


def evaluate_models(
    all_fold_predictions: Dict[str, Dict[str, Any]], output_csv_path: str
) -> None:
    """
    Aggregate per-fold evaluation metrics and save a summary CSV.

    Args:
        all_fold_predictions: Dict mapping fold/model name strings to fold result dicts.
        output_csv_path: Path where the summary CSV will be written.
    """
    results = []

    for model_name, fold_data in all_fold_predictions.items():
        if fold_data and "metrics" in fold_data:
            fold_results = {
                "Model Name": model_name,
                **fold_data["metrics"],
                "Eval Accuracy": fold_data["eval_acc"],
                "Eval Loss": fold_data["eval_loss"],
            }
            results.append(fold_results)

    if results:
        results_df = pd.DataFrame(results)

        avg_metrics = {
            "Model Name": "Average",
            "Precision": results_df["Precision"].mean(),
            "Recall": results_df["Recall"].mean(),
            "F2-Score": results_df["F2-Score"].mean(),
            "Eval Accuracy": results_df["Eval Accuracy"].mean(),
            "Eval Loss": results_df["Eval Loss"].mean(),
        }
        std_metrics = {
            "Model Name": "Std. Dev.",
            "Precision": results_df["Precision"].std(),
            "Recall": results_df["Recall"].std(),
            "F2-Score": results_df["F2-Score"].std(),
            "Eval Accuracy": results_df["Eval Accuracy"].std(),
            "Eval Loss": results_df["Eval Loss"].std(),
        }
        results_df = pd.concat(
            [results_df, pd.DataFrame([avg_metrics, std_metrics])], ignore_index=True
        )

        print(results_df.to_string(index=False, float_format="%.4f"))
        results_df.to_csv(output_csv_path, index=False)
    else:
        print("\nNo valid fold data was processed.")


# ── Artifact Saving ───────────────────────────────────────────────────────────


def save_predictions_from_detailed_results(
    detailed_results: List[Dict[str, Any]], filename: str
) -> None:
    """
    Write per-sample predictions to a CSV file for post-hoc analysis.

    Args:
        detailed_results: List of prediction dicts.
        filename: Path where the CSV will be written.
    """
    rows = [
        {
            "file_name": item["filename"],
            "true_label": item["true_label"],
            "prediction": item["pred_label"],
        }
        for item in detailed_results
    ]

    csv_headers = ["file_name", "true_label", "prediction"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(rows)



def save_metrics_to_csv(
    metrics_history: List[Dict[str, Any]],
    filename: str,
    detailed_results: List[Dict[str, Any]],
) -> None:
    """
    Save the per-epoch training history and the final evaluation metrics to a
    single CSV file.

    Args:
        metrics_history: List of per-epoch metric dicts.
        filename: Path where the CSV will be written.
        detailed_results: Per-sample prediction dicts.
    """
    if not metrics_history:
        print("Warning: metrics_history is empty. Cannot save metrics CSV.")
        return

    history_df = pd.DataFrame(metrics_history)

    final_metrics = calculate_classification_metrics(detailed_results)
    final_metrics_row = pd.DataFrame([{"epoch": "final_eval", **final_metrics}])

    full_metrics_df = pd.concat([history_df, final_metrics_row], ignore_index=True)
    full_metrics_df.to_csv(filename, index=False, float_format="%.4f")


def save_fold_artifacts(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    best_model_state: Dict[str, Any],
    detailed_results: List[Dict[str, Any]],
    metrics_history: List[Dict[str, Any]],
    experiment_name: str,
    fold: int,
    fold_output_dir: str,
    config: Dict[str, Any],
) -> None:
    """
    Generate and save all evaluation artifacts for a single completed fold.

    Args:
        model: Trained model (weights at the last epoch).
        eval_loader: DataLoader over the test split.
        best_model_state: Dict returned by ``run_training_loop``.
        detailed_results: Per-sample prediction records.
        metrics_history: Per-epoch metric dicts.
        experiment_name: Unique experiment identifier.
        fold: 1-based fold index.
        fold_output_dir: Directory where all artifacts are written.
        config: Experiment config dict.
    """

    cm_raw_path = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_cm_raw.png")
    cm_norm_path = os.path.join(
        fold_output_dir, f"{experiment_name}_fold{fold}_cm_normalized.png"
    )
    roc_path = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_roc_curve.png")
    predictions_csv = os.path.join(
        fold_output_dir, f"{experiment_name}_fold{fold}_predictions.csv"
    )
    metrics_csv = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_metrics.csv")
    last_model_path = os.path.join(
        fold_output_dir, f"{experiment_name}_fold{fold}_model_last.pth"
    )
    best_model_path = os.path.join(
        fold_output_dir, f"{experiment_name}_fold{fold}_model_best.pth"
    )

    # --- Artifacts from pre-computed detailed_results (no inference needed) ---
    true_labels = [r["true_label"] for r in detailed_results]
    predictions = [r["pred_label"] for r in detailed_results]
    cm = confusion_matrix(true_labels, predictions)

    plot_confusion_matrix(cm, config["CLASSES"], cm_raw_path, title=f"CM Raw (Fold {fold})")
    plot_confusion_matrix(
        cm, config["CLASSES"], cm_norm_path, normalize=True, title=f"CM Norm (Fold {fold})"
    )
    save_predictions_from_detailed_results(detailed_results, predictions_csv)

    # --- ROC curve requires re-running inference over the DataLoader ---
    plot_roc_curve(model, eval_loader, config["DEVICE"], config["CLASSES"], roc_path)

    # --- Metrics CSV: epoch history + final evaluation row ---
    save_metrics_to_csv(
        metrics_history=metrics_history,
        filename=metrics_csv,
        detailed_results=detailed_results,
    )

    # --- Model state files ---
    torch.save(model.state_dict(), last_model_path)

    if best_model_state:
        torch.save(best_model_state, best_model_path)
