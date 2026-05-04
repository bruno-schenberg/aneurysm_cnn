"""
plots.py

Plotting, metric calculation, and artifact-saving utilities for the training
engine. Called by the orchestrator at the end of each fold to produce a
complete record of training behaviour and evaluation results.

## Three function groups

Functions are organised by what they depend on:

  **Plotting** — visual outputs saved as PNG files. ``plot_confusion_matrix``
  takes a pre-computed NumPy array and has no external dependencies.
  ``plot_roc_curve`` is integration-level: it requires a live model and a
  DataLoader because it re-runs inference to collect class probabilities.

  **Metric Calculation** — pure numeric functions that operate on lists of
  prediction dicts. No GPU, no DataLoader, and no real NIfTI files are needed.
  These are the primary candidates for unit testing with synthetic data.

  **Artifact Saving** — functions that write files to disk (CSV, PNG, .pth).
  ``save_fold_artifacts`` is the top-level entry point that calls all other
  functions for a completed fold. It is integration-level because it calls
  ``plot_roc_curve``.

## Why separate pre-computed artifacts from inference-dependent artifacts?

Most artifacts — confusion matrices, prediction CSVs, metrics CSVs — are
derived from ``detailed_results`` and ``metrics_history``, which are already
in memory after training. Generating them requires no further GPU work. The
ROC curve is the exception: it needs continuous probability scores (not just
argmax predictions), so it must re-run a forward pass over the evaluation
loader. Keeping this separation explicit makes it easy to regenerate the
cheaper artifacts offline without a GPU.

## Why two model files per fold?

Each fold saves a ``_model_last.pth`` (weights at the final epoch) and a
``_model_best.pth`` (weights at the epoch with the highest validation F2).
In most runs the best epoch occurs well before the final epoch, especially
with early stopping. Saving both allows post-hoc comparison and ensures the
best-F2 model is always available for further evaluation even if the training
loop ran many epochs past its peak.
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
    Render a confusion matrix and save it as a PNG file.

    A confusion matrix shows the joint distribution of true and predicted
    labels. For binary classification the layout is:

        ┌──────────────────────┬──────────────────────┐
        │  True Neg (TN)       │  False Pos (FP)      │
        ├──────────────────────┼──────────────────────┤
        │  False Neg (FN)      │  True Pos (TP)       │
        └──────────────────────┴──────────────────────┘

    Row index = true class, column index = predicted class — the scikit-learn
    convention used by ``confusion_matrix``.

    **Row normalisation:** when ``normalize=True``, each row is divided by its
    sum, so cell (i, j) becomes the fraction of true-class-i samples predicted
    as class j. The diagonal then shows per-class recall: the fraction of each
    class that was correctly identified. This is useful for comparing class-
    level performance independent of class size.

    **Text contrast:** cell annotations are drawn in white when the cell value
    exceeds half the maximum value, and black otherwise. This prevents light
    text on a light background or dark text on a dark background, keeping
    the numbers readable across the full colour range.

    **Figure cleanup:** ``plt.close()`` is called after saving to release the
    figure's memory. Without this, matplotlib accumulates open figures across
    folds, which can exhaust memory on long runs.

    Args:
        cm: Confusion matrix array of shape ``(n_classes, n_classes)``,
            as returned by ``sklearn.metrics.confusion_matrix``.
        classes: Class label strings used for axis tick labels
            (e.g. ``['0', '1']`` for healthy / aneurysm).
        filename: Path where the PNG will be saved.
        normalize: If ``True``, row-normalise so each cell shows the fraction
            of true-class samples predicted in that column. If ``False``,
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
    Re-run inference to collect class probabilities, then plot and save a ROC curve.

    **What the ROC curve shows:** the Receiver Operating Characteristic curve
    plots the True Positive Rate (recall / sensitivity) against the False
    Positive Rate (1 - specificity) as the classification threshold is swept
    from 1 to 0. A random classifier follows the diagonal; a perfect classifier
    reaches the top-left corner. The Area Under the Curve (AUC) summarises
    this into a single number in [0, 1]. AUC = 0.5 is chance; AUC = 1.0 is
    perfect separation. Unlike F2, AUC is threshold-independent, making it
    a useful complement for comparing models.

    **Why re-run inference?** The training loop checkpoints argmax predictions
    (hard labels), not probability scores. The ROC curve requires continuous
    probability scores for the positive class to sweep the threshold. A second
    forward pass over the evaluation loader is therefore needed.

    **Probability extraction:** the model outputs raw logits of shape
    ``(batch, num_classes)``. For the standard two-class case
    (``outputs.shape[1] == 2``), ``softmax`` converts logits to probabilities
    and column index 1 gives the positive-class probability. If the model
    outputs a single scalar (e.g. a sigmoid head), it is squeezed directly.

    **Single-class guard:** ``roc_curve`` raises a ``ValueError`` if only one
    class is present in the labels (the curve is undefined). This can happen
    with very small or heavily imbalanced evaluation splits. The function
    checks for this before calling ``roc_curve`` and prints a warning instead
    of crashing.

    **Integration-level:** requires a live model, a DataLoader, and a device.
    Not covered by unit tests.

    Args:
        model: Trained model. This function sets it to ``eval()`` mode
            internally.
        dataloader: DataLoader over the evaluation split, yielding dicts with
            keys ``"image"`` and ``"label"``.
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
        classes: Class label strings — used in the plot title only.
        filename: Path where the PNG will be saved.
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
    Compute Precision, Recall, and F2-Score from a list of prediction records.

    All three metrics are computed with respect to the positive class
    (``pos_label=1``, i.e. aneurysm present). This is appropriate for binary
    detection: we want to know how well the model finds aneurysms, not how well
    it identifies healthy cases.

    **Why F2, not F1?**
    F-beta score is the harmonic mean of precision and recall, with recall
    weighted β² times more heavily than precision. With β=2, recall contributes
    four times as much as precision. In a clinical screening context, a false
    negative (missed aneurysm) carries far higher cost than a false positive
    (unnecessary follow-up scan), so the metric should penalise low recall
    more heavily than low precision. F2 encodes this asymmetry.

    **Precision, recall, and their trade-off:**
    Precision = TP / (TP + FP): of all predicted positives, what fraction
    are truly positive? Recall = TP / (TP + FN): of all true positives,
    what fraction did the model find? Raising the classification threshold
    increases precision but reduces recall; lowering it does the opposite.
    F2 rewards finding more true positives even at the cost of more false alarms.

    **``zero_division=0``:** if the model predicts no positive cases,
    precision is undefined (0/0). ``zero_division=0`` returns 0.0 in that case
    rather than raising a warning, which is the correct behaviour for early
    training epochs where the model may collapse to always predicting negative.

    **Error handling:** the function wraps computation in a try/except and
    returns NaN for all metrics on unexpected errors. This prevents a single
    bad fold from aborting the entire experiment summary.

    Unit-testable: no GPU, no DataLoader, and no real files required. Construct
    ``detailed_results`` directly with synthetic dicts for testing.

    Args:
        detailed_results: List of prediction dicts, each containing:
            - ``'true_label'`` (int): Ground-truth class (0 or 1).
            - ``'pred_label'`` (int): Model prediction (0 or 1).

    Returns:
        Dict with keys ``'Precision'``, ``'Recall'``, ``'F2-Score'``, each
        a float in [0, 1].

        Special cases:
        - Empty list → all values are ``float('nan')``.
        - Only one class present in true labels → all values are ``0.0``.
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
    all_fold_predictions: Dict[str, Dict[str, Any]], output_csv_path: str
) -> None:
    """
    Aggregate per-fold evaluation metrics and save a summary CSV.

    This function is called once at the end of a full experiment (after all
    folds have completed) by ``summarize_experiment_results`` in the
    orchestrator. It collects the pre-computed metrics from each fold result
    dict, prints a summary table, and writes it to a CSV that includes an
    Average row and a Std. Dev. row at the bottom.

    **Why include Std. Dev.?**
    In k-fold cross-validation, the mean metric across folds is the primary
    summary statistic, but the standard deviation is equally important: a model
    with F2 = 0.75 ± 0.02 is far more reliable than one with F2 = 0.75 ± 0.20.
    High variance across folds indicates that performance is sensitive to which
    cases end up in training vs validation — a sign of instability that the
    mean alone would hide. When ``USE_KFOLD=False`` (single split), both rows
    will show the metrics of the single fold, and Std. Dev. will be NaN,
    which is expected.

    **Pre-computed metrics:** this function reads ``fold_data["metrics"]``
    directly rather than re-computing metrics from raw predictions. The
    ``"metrics"`` dict was populated by ``calculate_classification_metrics``
    in the orchestrator after each fold's test evaluation. This avoids
    re-running the metric computation at summary time.

    Args:
        all_fold_predictions: Dict mapping fold/model name strings (e.g.
            ``"R3D18_B_lr1e4_fold1"``) to fold result dicts. Each value must
            contain:
            - ``'metrics'`` (dict): ``{'Precision', 'Recall', 'F2-Score'}``.
            - ``'eval_acc'`` (float): Test accuracy for the fold.
            - ``'eval_loss'`` (float): Test loss for the fold.
        output_csv_path: Path where the summary CSV will be written.
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
    Write per-sample predictions to a CSV file for post-hoc analysis.

    The output CSV has one row per evaluated sample and three columns:
    ``file_name``, ``true_label``, and ``prediction``. This file is the
    primary audit trail for understanding which specific cases the model got
    right or wrong. It can be loaded offline to recompute any metric,
    investigate systematic errors (e.g. all false negatives from the same
    scanner), or feed into error analysis tools.

    Uses Python's ``csv.DictWriter`` rather than pandas to avoid the overhead
    of constructing a full DataFrame for a simple sequential write.

    Unit-testable: pass any list of synthetic prediction dicts and a
    temporary path — no model, DataLoader, or GPU required.

    Args:
        detailed_results: List of prediction dicts, each containing:
            - ``'filename'`` (str): Base filename of the NIfTI scan.
            - ``'true_label'`` (int): Ground-truth class label.
            - ``'pred_label'`` (int): Model-predicted class label.
        filename: Path where the CSV will be written.
            The CSV columns are ``file_name``, ``true_label``, ``prediction``.
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
    Save the per-epoch training history and the final evaluation metrics to a
    single CSV file.

    The output has one row per training epoch (from ``metrics_history``) plus
    a ``final_eval`` row at the end. The epoch rows contain training and
    validation metrics logged during training (train loss, train acc, val loss,
    val acc, val F2, duration_sec). The ``final_eval`` row contains Precision, Recall, and
    F2-Score computed on the test set after the best-model checkpoint was
    restored.

    **Why combine training history and final eval in one file?**
    Keeping them together makes it easy to plot the full training curve and
    then compare it against final test performance in a single load. The
    ``epoch`` column uses the string ``"final_eval"`` for the last row so it
    is immediately distinguishable from integer epoch indices when the CSV is
    opened in a spreadsheet.

    Unit-testable: construct ``metrics_history`` and ``detailed_results``
    with synthetic dicts and pass a temporary path — no GPU or real data
    required.

    Args:
        metrics_history: List of per-epoch metric dicts as returned by
            ``run_training_loop``. Each dict should contain at least
            ``'epoch'``, ``'train_loss'``, ``'val_loss'``, ``'val_f2'``.
        filename: Path where the CSV will be written.
        detailed_results: Per-sample prediction dicts used to compute the
            ``final_eval`` row via ``calculate_classification_metrics``.
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

    This is the top-level entry point called by the orchestrator after each
    fold's training and test evaluation are complete. It delegates to the other
    functions in this module and produces the full set of outputs needed to
    review and compare experiment results.

    **Artifacts written to ``fold_output_dir``:**

      - ``{name}_fold{n}_cm_raw.png`` — confusion matrix with raw counts.
      - ``{name}_fold{n}_cm_normalized.png`` — row-normalised confusion matrix
        (diagonal = per-class recall).
      - ``{name}_fold{n}_roc_curve.png`` — ROC curve with AUC annotation.
        Requires re-running inference over ``eval_loader``.
      - ``{name}_fold{n}_predictions.csv`` — per-sample true/predicted labels,
        one row per evaluated case.
      - ``{name}_fold{n}_metrics.csv`` — per-epoch training history plus a
        ``final_eval`` row with test-set Precision, Recall, F2.
      - ``{name}_fold{n}_model_last.pth`` — model ``state_dict`` after the
        last training epoch.
      - ``{name}_fold{n}_model_best.pth`` — model ``state_dict`` at the epoch
        with the highest validation F2. Skipped if ``best_model_state`` is
        ``None`` (no epoch achieved F2 > 0).

    **Why save both last and best model weights?**
    The best-F2 checkpoint is what should be used for evaluation and
    deployment. The last-epoch checkpoint is saved as a safety net: if the
    best-F2 threshold was never exceeded (e.g. training collapsed early) and
    ``best_model_state`` is ``None``, the last-epoch weights are still
    available. It also allows diagnosing whether the model was still improving
    at the end of training.

    **Integration-level:** ``plot_roc_curve`` re-runs a forward pass over
    ``eval_loader``, requiring a live model and a device. All other artifacts
    are generated purely from the in-memory ``detailed_results`` and
    ``metrics_history`` without additional inference.

    Args:
        model: Trained model (weights at the last epoch). Used only by
            ``plot_roc_curve`` for inference.
        eval_loader: DataLoader over the test split. Used only by
            ``plot_roc_curve``.
        best_model_state: Dict returned by ``run_training_loop`` containing
            ``'state_dict'``, ``'best_epoch'``, and ``'best_val_f2'``.
            Pass ``None`` to skip saving the best-model file while still
            writing all other artifacts.
        detailed_results: Per-sample prediction records collected during test
            evaluation (from ``validate_one_epoch`` with
            ``return_details=True``).
        metrics_history: Per-epoch metric dicts from ``run_training_loop``.
        experiment_name: Unique experiment identifier (used in file names).
        fold: 1-based fold index (used in file names and print messages).
        fold_output_dir: Directory where all artifacts are written. Must
            already exist.
        config: Experiment config dict; must contain ``'CLASSES'`` (list of
            class label strings) and ``'DEVICE'`` (device string).
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
