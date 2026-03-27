"""
orchestrator.py

Manages the execution of a full k-fold cross-validation experiment: initialising
per-fold data loaders, reinitialising the model, running the training loop,
evaluating the best F2 checkpoint, and aggregating results across folds.
"""

import json
import os
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


# --- 1. Single Fold Runner ---


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
    Runs training and evaluation for a single cross-validation fold.
    Restores the F2-optimal checkpoint for final evaluation and saves all artifacts.

    Returns a FoldResult dict with per-sample predictions, classification metrics,
    eval accuracy, and eval loss.
    """
    print(f"\n--- Starting Fold {fold + 1}/{config['N_SPLITS']} ---")

    fold_output_dir = os.path.join(experiment_output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    print(f"Results for Fold {fold} will be saved to: {fold_output_dir}")

    # [Step 1] Initialise optimiser and loss criterion
    criterion_cls = nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

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

    # [Step 5] Save best-checkpoint metadata for artifact traceability (SC-004)
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

    # [Step 6] Save all evaluation artifacts (plots, CSVs, model weights)
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


# --- 2. Experiment Orchestration Helpers ---


def prepare_experiment_setup(config: Dict[str, Any]) -> tuple[str, Any]:
    """
    Validates the experiment configuration, creates the output directory, seeds
    global randomness, and returns the output directory path and a fold-split
    generator (yields raw file lists, not DataLoaders).
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
    print(
        f"  - Balancing: {config['balancing'] or 'None'}, "
        f"Folds: {config['N_SPLITS']}"
    )
    print(f"  - Results will be saved to: {experiment_output_dir}")
    print("=" * 80)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    set_seed(config["RANDOM_SEED"])
    os.makedirs(experiment_output_dir, exist_ok=True)

    all_data = get_data_list(data_dir)
    fold_splits = split_data(
        all_data=all_data,
        n_splits=config["N_SPLITS"],
        test_size=config["TEST_SPLIT_RATIO"],
        seed=config["RANDOM_SEED"],
    )

    return experiment_output_dir, fold_splits


def summarize_experiment_results(
    experiment_name: str,
    experiment_output_dir: str,
    all_fold_predictions: Dict[str, Any],
) -> None:
    """Aggregates per-fold results and saves the final cross-validation summary."""
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION COMPLETE FOR: {experiment_name}")
    print("\nEvaluating aggregate predictions...")
    summary_csv_path = os.path.join(
        experiment_output_dir, f"{experiment_name}_evaluation_summary.csv"
    )
    evaluate_models(all_fold_predictions, summary_csv_path)
    print("\n" + "=" * 80)


# --- 3. Main Experiment Orchestrator ---


def run_experiment(config: Dict[str, Any]) -> None:
    """
    Runs a full k-fold cross-validation experiment: splits data, builds per-fold
    DataLoaders, trains one model per fold, evaluates the best F2 checkpoint,
    and saves an aggregate summary.
    """
    experiment_output_dir, fold_splits = prepare_experiment_setup(config)

    balancing = config["balancing"]
    oversample = balancing == "oversampling"
    use_class_weights = balancing == "weighted_cost_function"

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
        )

        # Compute class weights from the training fold only to prevent label leakage
        aneurysm_positive_weight: Optional[torch.Tensor] = None
        if use_class_weights:
            aneurysm_positive_weight = get_class_weights(train_files)
            if aneurysm_positive_weight is not None:
                print(
                    f"Using class weights for fold {fold_idx}: "
                    f"{aneurysm_positive_weight.tolist()}"
                )

        print(f"\n[Fold {fold_idx + 1}] Setting up model: {config['model']}")
        model = get_model(
            model_name=config["model"], num_classes=len(config["CLASSES"])
        )
        model.to(config["DEVICE"])

        fold_results = run_one_fold(
            fold=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            config=config,
            weights_tensor=aneurysm_positive_weight,
            experiment_output_dir=experiment_output_dir,
        )

        fold_model_name = f"{config['name']}_fold{fold_idx}"
        all_fold_predictions[fold_model_name] = fold_results

        if config["QUICK_TEST"]:
            print("\n[INFO] QUICK_TEST is enabled. Breaking after one fold.")
            break

    summarize_experiment_results(
        experiment_name=config["name"],
        experiment_output_dir=experiment_output_dir,
        all_fold_predictions=all_fold_predictions,
    )
