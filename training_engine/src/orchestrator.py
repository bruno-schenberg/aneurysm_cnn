"""
orchestrator.py

Manages the execution of a full cross-validation experiment: initialising
per-fold DataLoaders, reinitialising the model for each fold, running the
training loop, restoring the best-F2 checkpoint for final evaluation, saving
all artifacts, and aggregating results across folds.

- The model is reinitialised from scratch for every fold so performance
  differences are attributable to the data split, not accumulated training state.
- ``del model`` + ``torch.cuda.empty_cache()`` between folds prevents temporarily
  holding two large models in VRAM simultaneously.
- ``HOLD_OUT_TEST_SET=True`` evaluates on the 20% hold-out test split (default);
  ``False`` reuses the validation split, which is useful for debugging on small datasets.
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
    parse_input_resolution,
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

    Steps: initialise AdamW + CrossEntropyLoss → run training loop →
    restore best-F2 checkpoint → evaluate on test or val set (unweighted loss)
    → save checkpoint metadata JSON → save fold artifacts.

    The eval criterion is always unweighted ``CrossEntropyLoss``, regardless of
    training balancing strategy, so the reported eval loss is an unbiased measure
    of prediction error.

    Args:
        fold: 1-based fold index, used in print messages and file names.
        train_loader: DataLoader for the training portion of this fold.
        val_loader: DataLoader for the validation portion of this fold.
        test_loader: DataLoader for the hold-out test set (identical across folds).
        model: Freshly initialised model, already moved to the correct device.
            Will be a ``MultiModalWrapper`` when ``USE_TABULAR=True``.
        config: Fully-formed experiment config dict.
        weights_tensor: Per-class loss weights tensor for weighted
            ``CrossEntropyLoss``, or ``None`` if not using weighted loss.
        experiment_output_dir: Root directory for this experiment's outputs.
            A ``fold_{n}`` subdirectory is created inside it.

    Returns:
        A FoldResult dict containing ``'predictions'``, ``'metrics'``,
        ``'eval_acc'``, and ``'eval_loss'``.
    """
    fold_output_dir = os.path.join(experiment_output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)

    # [Step 1] Initialise optimiser and loss criterion
    criterion_cls = nn.CrossEntropyLoss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config.get("WEIGHT_DECAY", 1e-4),
    )

    # [Step 2] Run the training loop; checkpoints at the highest val F2 epoch
    grad_accum_steps = config.get("GRAD_ACCUM_STEPS", 1)
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
        grad_accum_steps=grad_accum_steps,
        use_amp=config.get("USE_AMP", True),
    )
    # [Step 3] Restore AUC-optimal checkpoint for final evaluation
    if best_model_checkpoint is not None:
        model.load_state_dict(best_model_checkpoint["state_dict"])
    else:
        print(f"  Warning [Fold {fold}]: No checkpoint saved (all F2 = 0). Using final epoch weights.")

    # [Step 4] Evaluate on the appropriate set depending on experiment config
    if config.get("HOLD_OUT_TEST_SET", False):
        eval_loader = test_loader
        eval_set_name = "Test"
    else:
        eval_loader = val_loader
        eval_set_name = "Validation"

    eval_criterion = nn.CrossEntropyLoss()
    eval_loss, eval_acc, eval_f2, eval_auc, detailed_results = validate_one_epoch(
        model, eval_loader, eval_criterion, config["DEVICE"], return_details=True,
        use_amp=config.get("USE_AMP", True),
    )
    assert isinstance(detailed_results, list), (
        "validate_one_epoch did not return detailed results as a list."
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
                    "best_val_auc": best_model_checkpoint["best_val_auc"],
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

    total_epochs_run = metrics_history[-1]["epoch"] if metrics_history else 0
    best_epoch = best_model_checkpoint["best_epoch"] if best_model_checkpoint else "N/A"
    print(
        f"  Fold {fold} | Epochs: {total_epochs_run} (best: {best_epoch}) | "
        f"Time: {total_time:.1f} min | "
        f"Prec: {final_metrics['Precision']:.4f} | "
        f"Rec: {final_metrics['Recall']:.4f} | "
        f"F2: {final_metrics['F2-Score']:.4f} | "
        f"AUC: {eval_auc:.4f} | "
        f"Acc: {eval_acc:.4f} | Loss: {eval_loss:.4f}"
    )

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

    Called once per experiment before the fold loop so that setup-time failures
    (missing data path, bad config) surface before any GPU work begins. ``set_seed``
    is called here — not inside the fold loop — so the data split itself is
    reproducible and each fold produces a different k-fold partition.

    Args:
        config: Fully-formed experiment config dict as produced by
            ``prepare_experiment_configs`` in ``train_models.py``.

    Returns:
        Tuple of ``(experiment_output_dir, fold_splits)`` where
        ``fold_splits`` yields ``(fold_idx, train_files, val_files, test_files)``
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

    Thin wrapper around ``evaluate_models`` in ``plots.py``. The summary CSV
    contains one row per fold plus Average and Std. Dev. rows.

    Args:
        experiment_name: Unique experiment identifier, used to name the output CSV.
        experiment_output_dir: Root output directory for the experiment.
        all_fold_predictions: Dict mapping fold names to FoldResult dicts.
    """
    print(f"\nSUMMARY: {experiment_name}")
    summary_csv_path = os.path.join(
        experiment_output_dir, f"{experiment_name}_evaluation_summary.csv"
    )
    evaluate_models(all_fold_predictions, summary_csv_path)


# ----------------------------------------------------
# 3. Main Experiment Orchestrator
# ----------------------------------------------------


def run_experiment(config: Dict[str, Any]) -> None:
    """
    Run a complete cross-validation experiment end-to-end.

    Top-level entry point called by ``train_models.py`` for each experiment.
    Sequences setup, the fold loop, and post-experiment summarisation.

    The ``balancing`` config field resolves to two flags: ``oversample=True``
    attaches a ``WeightedRandomSampler`` to the training loader; ``use_class_weights=True``
    passes per-class loss weights to the criterion. Both are derived from the
    training fold only to prevent label leakage. If ``EXPERIMENT_BACKUP_DIR`` is
    set in the environment, the output directory is copied there after all folds complete.

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
    # Resolve INPUT_RESOLUTION string to a spatial-size tuple once for all folds
    spatial_size = parse_input_resolution(config.get("INPUT_RESOLUTION", "128x128x128"))

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
            spatial_size=spatial_size,
            cache_rate=config.get("CACHE_RATE", 1.0),
        )

        # Compute class weights from the training fold only to prevent label leakage
        aneurysm_positive_weight: Optional[torch.Tensor] = None
        if use_class_weights:
            aneurysm_positive_weight = get_class_weights(train_files)

        model = get_model(
            model_name=config["model"],
            num_classes=len(config["CLASSES"]),
            use_tabular=use_tabular,
            spatial_size=spatial_size,
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
