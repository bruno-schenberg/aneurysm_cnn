import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .data_preprocess import get_class_weights, get_folds
from .models import get_model
from .training import run_training_loop, validate_one_epoch
from .plots import (
    calculate_classification_metrics,
    evaluate_models,
    save_fold_artifacts,
)

# --- 1. Single Fold Runner ---
def run_one_fold(
    fold: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    config: Dict[str, Any],
    weights_tensor: Optional[torch.Tensor],
    experiment_output_dir: str
) -> Dict[str, Any]:
    """
    Runs the training and evaluation for a single fold of a cross-validation experiment.

    Returns:
        A dictionary containing the results for the fold (metrics, predictions, etc.).
    """
    print(f"\n--- Starting Fold {fold+1}/{config['N_SPLITS']} ---")

    # Create a dedicated output directory for this fold
    fold_output_dir = os.path.join(experiment_output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    print(f"Results for Fold {fold} will be saved to: {fold_output_dir}")

    criterion_cls = nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # [Step 3] Run training loop for the current fold
    print(f"\n[Fold {fold}] Running Training...")
    metrics_history, total_time, best_model_state = run_training_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion_cls=criterion_cls, optimizer=optimizer, device=config['DEVICE'],
        num_epochs=config['EPOCHS'], class_weights=weights_tensor
    )
    print(f"Fold {fold} training duration: {total_time:.2f} minutes.")

    # --- Load the best model state for test evaluation ---
    if best_model_state:
        print(f"\n[Fold {fold}] Loading best model state for final evaluation...")
        model.load_state_dict(best_model_state)
    else:
        print(f"\n[Fold {fold}] Warning: No best model state found. Evaluating using the last epoch's model.")

    # [Step 4] Final evaluation for this fold.
    # If HOLD_OUT_TEST_SET is True, we evaluate on the validation set for fold-level metrics.
    # Otherwise, we evaluate on the test set as before.
    eval_loader = val_loader if config.get('HOLD_OUT_TEST_SET', False) else test_loader
    eval_set_name = "Validation" if config.get('HOLD_OUT_TEST_SET', False) else "Test"

    print(f"\n[Fold {fold}] Final {eval_set_name} Set Evaluation...")
    eval_criterion = nn.CrossEntropyLoss() # Un-weighted for standard eval loss
    eval_loss, eval_acc, detailed_results = validate_one_epoch(
        model, eval_loader, eval_criterion, config['DEVICE'], return_details=True
    )
    assert isinstance(detailed_results, list), "validate_one_epoch did not return detailed results as a list."
    print(f"Fold {fold} {eval_set_name} Results (using best model):")
    print(f"  -> {eval_set_name} Loss: {eval_loss:.4f} | {eval_set_name} Acc: {eval_acc:.4f}")

    # [Step 5] Generate and save all evaluation artifacts for the fold
    # The loader passed here determines which dataset is used for plots and predictions.
    save_fold_artifacts(
        model=model,
        eval_loader=eval_loader,
        best_model_state=best_model_state,
        detailed_results=detailed_results,
        metrics_history=metrics_history,
        experiment_name=config['name'],
        fold=fold,
        fold_output_dir=fold_output_dir,
        config=config
    )

    # Return all necessary results for aggregation
    return {
        # Consistently name these keys for the summary table
        'eval_loss': eval_loss,
        'eval_acc': eval_acc,
        'detailed_results': detailed_results
    }

# --- 3. Single Experiment Runner ---
def run_experiment(config: Dict[str, Any]) -> None:
    """
    Sets up and runs a full K-Fold cross-validation experiment. It trains and
    evaluates a model for each fold, saving all results.
    """

    experiment_name = config['name']
    # Main output directory for this specific experiment
    EXPERIMENT_OUTPUT_DIR = os.path.join(config['OUTPUT_DIR'], experiment_name)

    data_dir = config['data_path']
    model_name = config['model']
    balancing = config['balancing']

    # --- Parse balancing strategy ---
    oversample = (balancing == "oversampling")
    use_class_weights = (balancing == "weighted_cost_function")

    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT: {config['name']}")
    print(f"  - Model: {config['model']}, Epochs: {config['EPOCHS']}, BS: {config['BATCH_SIZE']}, LR: {config['LEARNING_RATE']}")
    print(f"  - Balancing: {balancing if balancing else 'None'}, Folds: {config['N_SPLITS']}")
    print(f"  - Results will be saved to: {EXPERIMENT_OUTPUT_DIR}")
    print("="*80)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found for experiment {experiment_name}: {data_dir}")

    print("\nInitializing K-Fold data generator...")
    fold_generator = get_folds(
        data_dir=data_dir,
        n_splits=config['N_SPLITS'],
        batch_size=config['BATCH_SIZE'],
        val_batch_size=config['VAL_BATCH_SIZE'],
        test_size=config['TEST_SPLIT_RATIO'],
        seed=config['RANDOM_SEED'],
        oversample=oversample
    )

    weights_tensor = None
    if use_class_weights:
        weights_tensor = get_class_weights(data_dir, config['CLASSES'])
        if weights_tensor is not None:
            print(f"Using class weights in loss function: {weights_tensor.tolist()}")

    all_fold_predictions = {} # To store detailed results for final evaluation

    # --- Loop over all folds ---
    for fold, train_loader, val_loader, test_loader in fold_generator: # type: ignore
        # Re-initialize model for each fold for a fair evaluation
        print(f"\n[Fold {fold+1}] Setting up model: {model_name}")
        model = get_model(model_name=model_name, num_classes=len(config['CLASSES']))
        model.to(config['DEVICE'])

        fold_results = run_one_fold(
            fold=fold, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, model=model, config=config,
            weights_tensor=weights_tensor,
            experiment_output_dir=EXPERIMENT_OUTPUT_DIR
        )

        # Store results for final aggregation
        final_metrics = calculate_classification_metrics(fold_results['detailed_results'])
        fold_model_name = f"{experiment_name}_fold{fold}"
        all_fold_predictions[fold_model_name] = {
            'predictions': fold_results['detailed_results'],
            'metrics': final_metrics,
            'eval_acc': fold_results['eval_acc'],
            'eval_loss': fold_results['eval_loss']
        }

        # If quick test is enabled, break after the first fold
        if config['QUICK_TEST']:
            print("\n[INFO] QUICK_TEST is enabled. Breaking after one fold.")
            break

    # --- After all folds are complete ---
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION COMPLETE FOR: {experiment_name}")

    print("\nEvaluating aggregate predictions...")
    summary_csv_path = os.path.join(EXPERIMENT_OUTPUT_DIR, f"{experiment_name}_evaluation_summary.csv")
    evaluate_models(all_fold_predictions, summary_csv_path)

    print("\n" + "="*80)