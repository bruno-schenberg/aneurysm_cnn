"""
Plotting, metric calculation, and artifact-saving utilities for the training engine.

Functions are grouped into three sections:
- **Plotting**: visual outputs (confusion matrices, ROC curves).
- **Metric Calculation**: pure numeric functions testable without GPU or real data.
- **Artifact Saving**: functions that write files to disk (CSVs, model weights).

The ``save_fold_artifacts`` and ``plot_roc_curve`` functions require a live model
and DataLoader; they are integration-level and are not covered by unit tests.
All other functions are unit-testable with synthetic data and temporary paths.
"""

import csv
import itertools
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
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
    Plot a confusion matrix and save it as a PNG file.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes), as returned
            by ``sklearn.metrics.confusion_matrix``.
        classes: List of class label strings used for axis tick labels
            (e.g. ``['0', '1']`` for healthy / aneurysm).
        filename: Absolute or relative path where the PNG will be saved.
        normalize: If ``True``, each row is divided by its sum so cell values
            represent per-class recall rates (row-normalised). If ``False``,
            raw counts are shown.
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
    print(f"Confusion matrix saved to {filename}")


def plot_roc_curve(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    classes: List[str],
    filename: str,
) -> None:
    """
    Compute and plot a Receiver Operating Characteristic (ROC) curve.

    **Integration-level**: requires a live model, a DataLoader, and a device.
    Not covered by unit tests — use ``calculate_classification_metrics`` for
    offline metric evaluation.

    The function iterates over the DataLoader, collects model probabilities for
    the positive class (class index 1), and plots FPR vs TPR with AUC annotation.
    If only one class is present in the evaluation labels, no file is written and
    a warning is printed instead.

    Args:
        model: Trained model in eval mode, or to be set to eval mode here.
        dataloader: DataLoader yielding dicts with keys ``"image"`` and ``"label"``.
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
        classes: Class label strings (used for the plot title only).
        filename: Absolute or relative path where the PNG will be saved.
    """
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []

    progress_bar = tqdm.tqdm(dataloader, desc="Calculating ROC", unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
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
    print(f"ROC curve saved to {filename}")


# ── Metric Calculation ────────────────────────────────────────────────────────

def calculate_classification_metrics(
    detailed_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F2-Score from a list of prediction records.

    F2-Score (β=2) weights recall twice as heavily as precision, reflecting the
    clinical priority of minimising missed aneurysms (false negatives) over
    minimising false positives.

    Unit-testable: no GPU, no DataLoader, and no real files required. Construct
    ``detailed_results`` directly with synthetic dicts for testing.

    Args:
        detailed_results: List of prediction dicts. Each dict must contain:
            - ``'true_label'`` (int): Ground-truth class (0 or 1).
            - ``'pred_label'`` (int): Model prediction (0 or 1).

    Returns:
        Dict with keys ``'Precision'``, ``'Recall'``, and ``'F2-Score'``, each
        a float in [0, 1].

        Special cases:
        - Empty list → all values are ``float('nan')``.
        - Only one class in true labels → all values are ``0.0``.
        - Any unexpected exception → all values are ``float('nan')``.
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
    all_fold_predictions: Dict[str, List[Dict[str, Any]]], output_csv_path: str
) -> None:
    """
    Aggregate per-fold metrics across all folds and save a summary CSV.

    Iterates over ``all_fold_predictions``, calculates Precision, Recall, and
    F2-Score for each fold, appends aggregate Average and Std. Dev. rows, and
    saves the result to ``output_csv_path``.

    Args:
        all_fold_predictions: Dict mapping fold/model name strings to fold result
            dicts. Each fold result dict is expected to contain:
            - ``'metrics'`` (dict): Pre-computed ``{'Precision', 'Recall', 'F2-Score'}``.
            - ``'eval_acc'`` (float): Evaluation accuracy for the fold.
            - ``'eval_loss'`` (float): Evaluation loss for the fold.
        output_csv_path: Path where the summary CSV file will be written.
    """
    print(f"\nEvaluating predictions for {len(all_fold_predictions)} folds...")
    results = []

    for model_name, fold_data in all_fold_predictions.items():
        if fold_data and "metrics" in fold_data:
            print(f"--- Aggregating results for **{model_name}** ---")
            fold_results = {
                "Model Name": model_name,
                **fold_data["metrics"],
                "Eval Accuracy": fold_data["eval_acc"],
                "Eval Loss": fold_data["eval_loss"],
            }
            results.append(fold_results)
            print(
                f"  Eval Acc: {fold_results['Eval Accuracy']:.4f}, "
                f"Eval Loss: {fold_results['Eval Loss']:.4f}, "
                f"F2-Score: {fold_results['F2-Score']:.4f}"
            )

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

        print("\n" + "=" * 50)
        print("FINAL MODEL EVALUATION SUMMARY")
        print("=" * 50)
        print(results_df.to_string(index=False, float_format="%.4f"))
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nEvaluation summary saved to: {output_csv_path}")
    else:
        print("\nNo valid fold data was processed.")


# ── Artifact Saving ───────────────────────────────────────────────────────────

def save_predictions_from_detailed_results(
    detailed_results: List[Dict[str, Any]], filename: str
) -> None:
    """
    Write per-sample predictions to a CSV file.

    Unit-testable: pass any list of synthetic prediction dicts and a temporary
    path — no model, DataLoader, or GPU is required.

    Args:
        detailed_results: List of prediction dicts. Each dict must contain:
            - ``'filename'`` (str): Base filename of the NIfTI scan.
            - ``'true_label'`` (int): Ground-truth class label.
            - ``'pred_label'`` (int): Model-predicted class label.
        filename: Absolute or relative path where the CSV will be written.
            The CSV has columns ``file_name``, ``true_label``, ``prediction``.
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

    print(f"Test set predictions saved to {filename}")


def save_metrics_to_csv(
    metrics_history: List[Dict[str, Any]],
    filename: str,
    detailed_results: List[Dict[str, Any]],
) -> None:
    """
    Save the per-epoch training/validation history plus final evaluation metrics to CSV.

    Produces a single CSV containing one row per training epoch (from
    ``metrics_history``) followed by a ``final_eval`` row with Precision, Recall,
    and F2-Score computed from ``detailed_results``.

    Unit-testable: construct ``metrics_history`` and ``detailed_results`` directly
    with synthetic dicts and pass a temporary path — no GPU or real data required.

    Args:
        metrics_history: List of per-epoch metric dicts (e.g. train loss, val loss,
            val accuracy). Each dict should contain at least an ``'epoch'`` key.
        filename: Absolute or relative path where the CSV will be written.
        detailed_results: List of prediction dicts used to compute the final
            evaluation metrics (see ``calculate_classification_metrics``).
    """
    if not metrics_history:
        print("Warning: metrics_history is empty. Cannot save metrics CSV.")
        return

    history_df = pd.DataFrame(metrics_history)

    final_metrics = calculate_classification_metrics(detailed_results)
    final_metrics_row = pd.DataFrame([{"epoch": "final_eval", **final_metrics}])

    full_metrics_df = pd.concat([history_df, final_metrics_row], ignore_index=True)
    full_metrics_df.to_csv(filename, index=False, float_format="%.4f")
    print(f"Metrics saved to {filename}")


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

    Artifacts written to ``fold_output_dir``:
    - ``{name}_fold{n}_cm_raw.png`` — raw count confusion matrix.
    - ``{name}_fold{n}_cm_normalized.png`` — row-normalised confusion matrix.
    - ``{name}_fold{n}_roc_curve.png`` — ROC curve with AUC (requires DataLoader).
    - ``{name}_fold{n}_predictions.csv`` — per-sample predictions.
    - ``{name}_fold{n}_metrics.csv`` — per-epoch history + ``final_eval`` row.
    - ``{name}_fold{n}_model_last.pth`` — ``state_dict`` after the final epoch.
    - ``{name}_fold{n}_model_best.pth`` — ``state_dict`` at the best F2-Score epoch
      (skipped if ``best_model_state`` is ``None``).

    **Integration-level**: ``plot_roc_curve`` requires a live model and DataLoader.
    All other artifacts are generated from pre-computed ``detailed_results`` and
    ``metrics_history`` without re-running inference.

    Args:
        model: Trained model (used only for ROC curve inference).
        eval_loader: DataLoader over the evaluation split (used only for ROC curve).
        best_model_state: ``state_dict`` captured at the epoch with the highest
            F2-Score during training. Pass ``None`` to skip saving the best-model
            file while still writing all other artifacts.
        detailed_results: Per-sample prediction records for the fold.
        metrics_history: Per-epoch metric dicts for the fold.
        experiment_name: Unique experiment identifier (used in file names).
        fold: Fold index (used in file names).
        fold_output_dir: Directory where all artifacts are written.
        config: Experiment config dict; must contain keys ``'CLASSES'`` and
            ``'DEVICE'`` (used by ``plot_confusion_matrix`` and ``plot_roc_curve``).
    """
    print(f"\n[Fold {fold}] Generating and saving evaluation artifacts...")

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
    print(f"Last model for fold {fold} saved to {last_model_path}")

    if best_model_state:
        torch.save(best_model_state, best_model_path)
        print(f"Best model for fold {fold} saved to {best_model_path}")
