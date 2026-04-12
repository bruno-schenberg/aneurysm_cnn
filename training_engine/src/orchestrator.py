"""
orchestrator.py

Manages the execution of a full cross-validation experiment: initialising
per-fold DataLoaders, reinitialising the model for each fold, running the
training loop, restoring the best-F2 checkpoint for final evaluation, saving
all artifacts, and aggregating results across folds.

## Role in the pipeline

The orchestrator is the conductor that wires together every other module in
the training engine. It does not implement any training logic itself — that
lives in ``training.py``. Instead it sequences the pipeline:

    prepare_experiment_setup          ← validate config, seed RNG, split data
           ↓ (per fold)
    build_dataloaders                 ← wrap file lists in DataLoaders
    get_class_weights                 ← compute per-fold loss weights
    get_model → model.to(device)      ← fresh model for every fold
    run_one_fold
        ├─ run_training_loop          ← train + checkpoint best F2
        ├─ model.load_state_dict      ← restore best checkpoint
        ├─ validate_one_epoch         ← final test/val evaluation
        └─ save_fold_artifacts        ← plots, CSVs, model weights
           ↓ (after all folds)
    summarize_experiment_results      ← aggregate CSV across folds

## Why reinitialise the model for every fold?

Each fold must start from the same random initialisation to ensure that
performance differences between folds are attributable to the data split,
not to accumulated training history from previous folds. ``get_model`` is
called inside the fold loop so every fold gets a freshly initialised model
rather than continuing from where the previous fold left off.

## Why delete the model and empty the CUDA cache between folds?

A 3D CNN can occupy several gigabytes of GPU memory. Holding the previous
fold's model in memory while initialising the next fold's model would require
temporarily fitting two models simultaneously, which often exceeds available
VRAM. ``del model`` drops the Python reference, making the object eligible for
garbage collection, and ``torch.cuda.empty_cache()`` tells the CUDA allocator
to release any cached (but unused) memory back to the OS so the next fold
starts with a clean allocation.

## Evaluation set: test vs validation

The ``HOLD_OUT_TEST_SET`` config flag controls which DataLoader is used for
final fold evaluation. When ``True`` (the default), the 20% hold-out test set
is used — this is the correct setting for reporting results. When ``False``,
the validation split is reused for evaluation. The latter is primarily useful
for debugging on small synthetic datasets where a three-way split would leave
too few samples in each subset.
"""

import json
import os
import shutil
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .data_preprocess import (
    build_dataloaders,
    get_class_weights,
    get_data_list,
    set_seed,
    split_data,
)
from .models import get_model
from .plots import (
    calculate_classification_metrics,
    evaluate_models,
    save_fold_artifacts,
)
from .training import run_training_loop, validate_one_epoch


# ----------------------------------------------------
# 1. Single Fold Runner
# ----------------------------------------------------


def run_one_fold(
    fold: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    config: Dict[str, Any],
    weights_tensor: Optional[torch.Tensor],
    experiment_output_dir: str,
) -> Dict[str, Any]:
    """
    Execute one complete fold: train, restore best checkpoint, evaluate, save
    artifacts, and return a result dict for aggregation.

    This function is the innermost unit of the experiment loop. It is called
    once per fold by ``run_experiment`` with a freshly initialised model and
    pre-built DataLoaders. All six steps are sequential and each depends on
    the output of the previous one.

    **Step 1 — Optimiser and loss criterion:**
    ``AdamW`` is used with the learning rate and weight decay from the config.
    AdamW decouples weight decay from the gradient update, which makes the
    decay coefficient behave more predictably than the weight decay in
    standard Adam. ``criterion_cls`` is passed as a *class* (not an instance)
    to ``run_training_loop``, which instantiates it internally with the correct
    device placement for the class weights tensor. See ``training.py`` for the
    rationale.

    **Step 2 — Training loop:**
    ``run_training_loop`` runs all epochs, checkpointing at the best val F2.
    It returns the full per-epoch metrics history and the best checkpoint dict
    (or ``None`` if no epoch achieved F2 > 0).

    **Step 3 — Checkpoint restoration:**
    After training, the model's weights are at the *last* epoch, not the best
    epoch. The best-F2 checkpoint is explicitly loaded back with
    ``load_state_dict`` before evaluation. This is a critical step: evaluating
    the last-epoch weights instead of the best-F2 weights would measure a
    different model and likely produce worse (and misleading) results. If no
    checkpoint was saved (F2 was always 0), the final-epoch weights are used
    and a warning is printed.

    **Step 4 — Evaluation set selection:**
    ``HOLD_OUT_TEST_SET=True`` evaluates on the held-out test split (20% of
    total data, never seen during training or hyperparameter decisions).
    ``HOLD_OUT_TEST_SET=False`` reuses the validation split. The eval criterion
    is always an *unweighted* ``CrossEntropyLoss``, regardless of whether
    class weights were used during training. This ensures the reported eval
    loss is a fair, unbiased measure of prediction error rather than a
    weighted loss that depends on the chosen balancing strategy.

    **Step 5 — Checkpoint metadata JSON:**
    A small JSON file recording the best epoch number and its val F2 is saved
    alongside the fold artifacts. This provides a human-readable record of
    which training epoch was selected, making it easy to check whether training
    had converged or was still improving when the best model was captured.

    **Step 6 — Artifact saving:**
    Delegates to ``save_fold_artifacts`` in ``plots.py``, which generates
    confusion matrices, a ROC curve, prediction and metrics CSVs, and model
    weight files. Only the ``state_dict`` component of the best checkpoint is
    passed (not the full dict) so the saved ``.pth`` file contains only the
    weights and can be loaded directly with ``model.load_state_dict``.

    **Tabular (multimodal) support:** When ``config["USE_TABULAR"]`` is
    ``True``, the DataLoaders produced by ``prepare_experiment_setup`` include
    a ``"tabular"`` key in each batch. The model passed in will already be a
    ``MultiModalWrapper`` (constructed by ``get_model`` in ``run_experiment``).
    The training and validation functions detect the ``"tabular"`` key at
    runtime and adjust the forward call automatically — no changes are needed
    here to support the multimodal path.

    Args:
        fold: 1-based fold index, used in print messages and file names.
        train_loader: DataLoader for the training portion of this fold.
        val_loader: DataLoader for the validation portion of this fold.
        test_loader: DataLoader for the hold-out test set (identical across
            all folds — it is not a per-fold split).
        model: Freshly initialised model, already moved to the correct device.
            Will be a ``MultiModalWrapper`` when ``USE_TABULAR=True``.
        config: Fully-formed experiment config dict.
        weights_tensor: Per-class loss weights tensor for weighted
            ``CrossEntropyLoss``, or ``None`` if balancing is not
            ``'weighted_cost_function'``.
        experiment_output_dir: Root directory for this experiment's outputs.
            A ``fold_{n}`` subdirectory is created inside it.

    Returns:
        A FoldResult dict containing:
        - ``'predictions'``: List of per-sample prediction dicts
          (filename, true_label, pred_label).
        - ``'metrics'``: Dict of ``{'Precision', 'Recall', 'F2-Score'}``
          computed on the evaluation set.
        - ``'eval_acc'``: Scalar evaluation accuracy.
        - ``'eval_loss'``: Scalar evaluation loss (unweighted).
    """
    total_folds = config["N_SPLITS"] if config.get("USE_KFOLD", True) else 1
    print(f"\n--- Starting Fold {fold}/{total_folds} ---")

    fold_output_dir = os.path.join(experiment_output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    print(f"Results for Fold {fold} will be saved to: {fold_output_dir}")

    # [Step 1] Initialise optimiser and loss criterion
    criterion_cls = nn.CrossEntropyLoss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config.get("WEIGHT_DECAY", 1e-4),
    )

    # [Step 2] Run the training loop; checkpoints at the highest val F2 epoch
    metrics_history, total_time, best_model_checkpoint = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_cls=criterion_cls,
        optimizer=optimizer,
        device=config["DEVICE"],
        num_epochs=config["EPOCHS"],
        class_weights=weights_tensor,
        patience=config.get("EARLY_STOPPING_PATIENCE", 0),
    )
    print(f"Fold {fold} training duration: {total_time:.2f} minutes.")

    # [Step 3] Restore F2-optimal checkpoint for final evaluation
    if best_model_checkpoint is not None:
        print(
            f"\n[Fold {fold}] Loading F2-optimal checkpoint "
            f"(epoch {best_model_checkpoint['best_epoch']}, "
            f"val F2 = {best_model_checkpoint['best_val_f2']:.4f})..."
        )
        model.load_state_dict(best_model_checkpoint["state_dict"])
    else:
        print(
            f"\n[Fold {fold}] Warning: No checkpoint saved (all F2 values were 0). "
            "Evaluating with final epoch weights."
        )

    # [Step 4] Evaluate on the appropriate set depending on experiment config
    if config.get("HOLD_OUT_TEST_SET", False):
        eval_loader = test_loader
        eval_set_name = "Test"
    else:
        eval_loader = val_loader
        eval_set_name = "Validation"

    print(f"\n[Fold {fold}] Final {eval_set_name} Set Evaluation...")
    eval_criterion = nn.CrossEntropyLoss()
    eval_loss, eval_acc, eval_f2, detailed_results = validate_one_epoch(
        model, eval_loader, eval_criterion, config["DEVICE"], return_details=True
    )
    assert isinstance(detailed_results, list), (
        "validate_one_epoch did not return detailed results as a list."
    )
    print(
        f"  -> Fold {eval_set_name} Results: "
        f"Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f} | F2: {eval_f2:.4f}"
    )

    # [Step 5] Save best-checkpoint metadata for artifact traceability
    if best_model_checkpoint is not None:
        checkpoint_metadata_path = os.path.join(
            fold_output_dir, "best_checkpoint_metadata.json"
        )
        with open(checkpoint_metadata_path, "w") as f:
            json.dump(
                {
                    "best_epoch": best_model_checkpoint["best_epoch"],
                    "best_val_f2": best_model_checkpoint["best_val_f2"],
                },
                f,
                indent=2,
            )

    # [Step 6] Save all evaluation artifacts (plots, CSVs, model weights).
    # Only the state_dict is passed — not the full checkpoint dict — so the
    # saved .pth file can be loaded directly with model.load_state_dict().
    best_model_state_dict = (
        best_model_checkpoint["state_dict"]
        if best_model_checkpoint is not None
        else None
    )
    save_fold_artifacts(
        model=model,
        eval_loader=eval_loader,
        best_model_state=best_model_state_dict,
        detailed_results=detailed_results,
        metrics_history=metrics_history,
        experiment_name=config["name"],
        fold=fold,
        fold_output_dir=fold_output_dir,
        config=config,
    )

    final_metrics = calculate_classification_metrics(detailed_results)

    return {
        "predictions": detailed_results,
        "metrics": final_metrics,
        "eval_acc": eval_acc,
        "eval_loss": eval_loss,
    }


# ----------------------------------------------------
# 2. Experiment Orchestration Helpers
# ----------------------------------------------------


def prepare_experiment_setup(config: Dict[str, Any]) -> tuple[str, Any]:
    """
    Validate the experiment configuration, seed the RNG, create the output
    directory, and return a fold-split generator.

    This function runs once per experiment, before the fold loop. It is
    separated from ``run_experiment`` so that setup-time failures (missing
    data path, bad config) surface before any GPU work begins.

    **Why seed here, not inside the fold loop?**
    ``set_seed`` is called once at setup time so that the data split itself
    is reproducible. If it were called at the start of each fold, the split
    would be re-seeded to the same state every iteration, producing identical
    folds rather than different k-fold partitions. The split generator
    (``fold_splits``) is created after seeding, so its entire sequence is
    deterministic from this single seed call.

    **Generator, not a list:**
    ``split_data`` returns a generator. The fold loop in ``run_experiment``
    advances it one fold at a time, so only one fold's file lists are in
    memory at once. This matters for large datasets but is mostly a style
    choice here since the file lists are small.

    Args:
        config: Fully-formed experiment config dict as produced by
            ``prepare_experiment_configs`` in ``train_models.py``.

    Returns:
        Tuple of ``(experiment_output_dir, fold_splits)`` where
        ``experiment_output_dir`` is the path to the experiment's output
        directory (created if it does not exist) and ``fold_splits`` is a
        generator yielding ``(fold_idx, train_files, val_files, test_files)``
        tuples.

    Raises:
        FileNotFoundError: If the dataset directory specified in
            ``config['data_path']`` does not exist.
    """
    experiment_name = config["name"]
    experiment_output_dir = os.path.join(config["OUTPUT_DIR"], experiment_name)
    data_dir = config["data_path"]

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(
        f"  - Model: {config['model']}, Epochs: {config['EPOCHS']}, "
        f"BS: {config['BATCH_SIZE']}, LR: {config['LEARNING_RATE']}"
    )
    use_kfold = config.get("USE_KFOLD", True)
    val_pct = int(config.get("VAL_SPLIT_RATIO", 0.30) * 100)
    folds_display = config["N_SPLITS"] if use_kfold else f"1 (single {100 - val_pct}/{val_pct} split)"
    print(
        f"  - Balancing: {config['balancing'] or 'None'}, "
        f"Folds: {folds_display}"
    )
    print(f"  - Results will be saved to: {experiment_output_dir}")
    print("=" * 80)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    set_seed(config["RANDOM_SEED"])
    os.makedirs(experiment_output_dir, exist_ok=True)

    tabular_csv = config.get("TABULAR_CSV")
    all_data = get_data_list(data_dir, tabular_csv=tabular_csv)
    fold_splits = split_data(
        all_data=all_data,
        n_splits=config["N_SPLITS"],
        test_size=config["TEST_SPLIT_RATIO"],
        seed=config["RANDOM_SEED"],
        use_kfold=config.get("USE_KFOLD", True),
        val_split_ratio=config.get("VAL_SPLIT_RATIO", 0.30),
    )

    return experiment_output_dir, fold_splits


def summarize_experiment_results(
    experiment_name: str,
    experiment_output_dir: str,
    all_fold_predictions: Dict[str, Any],
) -> None:
    """
    Aggregate per-fold results and save the experiment-level summary CSV.

    Thin wrapper around ``evaluate_models`` in ``plots.py``. Called once after
    all folds complete. The summary CSV contains one row per fold plus Average
    and Std. Dev. rows; see ``evaluate_models`` for details.

    Args:
        experiment_name: Unique experiment identifier, used to name the
            output CSV file.
        experiment_output_dir: Root output directory for the experiment.
        all_fold_predictions: Dict mapping fold names to FoldResult dicts,
            as accumulated by ``run_experiment``.
    """
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION COMPLETE FOR: {experiment_name}")
    print("\nEvaluating aggregate predictions...")
    summary_csv_path = os.path.join(
        experiment_output_dir, f"{experiment_name}_evaluation_summary.csv"
    )
    evaluate_models(all_fold_predictions, summary_csv_path)
    print("\n" + "=" * 80)


# ----------------------------------------------------
# 3. Main Experiment Orchestrator
# ----------------------------------------------------


def run_experiment(config: Dict[str, Any]) -> None:
    """
    Run a complete cross-validation experiment end-to-end.

    This is the top-level entry point called by ``train_models.py`` for each
    experiment defined in ``experiments.json``. It sequences setup, the fold
    loop, and post-experiment summarisation.

    **Balancing strategy resolution:**
    The ``balancing`` config field is a string (``'weighted_cost_function'``,
    ``'oversampling'``, or ``'none'``) that resolves to two independent
    boolean flags:

      - ``oversample=True``: passed to ``build_dataloaders``, which attaches
        a ``WeightedRandomSampler`` to the training loader so each class
        appears equally often per epoch.
      - ``use_class_weights=True``: triggers a call to ``get_class_weights``
        on the training fold, producing a weight tensor passed to the loss
        function so misclassifying the minority class is penalised more
        heavily.

    Both flags are mutually exclusive in practice (the config only sets one
    strategy per experiment), but the code does not enforce this — they could
    theoretically be combined.

    **Class weight computation inside the fold loop:**
    ``get_class_weights`` is called *after* ``split_data`` and *inside* the
    fold loop, using only ``train_files`` for the current fold. This prevents
    label leakage: the loss weighting is derived solely from the training
    distribution of each fold, not from the full dataset.

    **Optional backup:**
    If the ``EXPERIMENT_BACKUP_DIR`` environment variable is set, the entire
    experiment output directory is copied there after all folds complete using
    ``shutil.copytree``. This is used on HPC clusters where the local
    ``experiments/`` directory may be purged after a job finishes.

    Args:
        config: Fully-formed experiment config dict as returned by
            ``prepare_experiment_configs`` in ``train_models.py``.
    """
    experiment_output_dir, fold_splits = prepare_experiment_setup(config)

    balancing = config["balancing"]
    oversample = balancing == "oversampling"
    use_class_weights = balancing == "weighted_cost_function"
    use_tabular = config.get("USE_TABULAR", False)

    all_fold_predictions: Dict[str, Any] = {}

    for fold_idx, train_files, val_files, test_files in fold_splits:
        train_loader, val_loader, test_loader = build_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=config["BATCH_SIZE"],
            val_batch_size=config["VAL_BATCH_SIZE"],
            seed=config["RANDOM_SEED"],
            oversample=oversample,
            use_tabular=use_tabular,
        )

        # Compute class weights from the training fold only to prevent label leakage
        aneurysm_positive_weight: Optional[torch.Tensor] = None
        if use_class_weights:
            aneurysm_positive_weight = get_class_weights(train_files)
            if aneurysm_positive_weight is not None:
                print(
                    f"Using class weights for fold {fold_idx + 1}: "
                    f"{aneurysm_positive_weight.tolist()}"
                )

        print(f"\n[Fold {fold_idx + 1}] Setting up model: {config['model']}")
        model = get_model(
            model_name=config["model"],
            num_classes=len(config["CLASSES"]),
            use_tabular=use_tabular,
        )
        model.to(config["DEVICE"])

        fold_results = run_one_fold(
            fold=fold_idx + 1,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            config=config,
            weights_tensor=aneurysm_positive_weight,
            experiment_output_dir=experiment_output_dir,
        )

        # Release GPU memory before the next fold allocates a fresh model
        del model
        if config["DEVICE"] == "cuda":
            torch.cuda.empty_cache()

        fold_model_name = f"{config['name']}_fold{fold_idx + 1}"
        all_fold_predictions[fold_model_name] = fold_results

    summarize_experiment_results(
        experiment_name=config["name"],
        experiment_output_dir=experiment_output_dir,
        all_fold_predictions=all_fold_predictions,
    )

    backup_dir = os.environ.get("EXPERIMENT_BACKUP_DIR")
    if backup_dir:
        dest = os.path.join(backup_dir, config["name"])
        shutil.copytree(experiment_output_dir, dest, dirs_exist_ok=True)
        print(f"[INFO] Results backed up to: {dest}")
