# @title
import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Any, Tuple
import tqdm
import numpy as np # Import numpy for weight calculation

# --- 1. Import Modules ---
from data_preprocess import get_data_loaders
from plots import plot_confusion_matrix, plot_roc_curve, save_test_predictions_to_csv, save_metrics_to_csv
from training import run_training_loop, validate_one_epoch
from models import get_model

# --- 2. Configuration (Simplified and Static) ---
class StaticConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    RANDOM_SEED = 42
    CLASSES = ['0', '1'] # 0: healthy, 1: aneurysm

    # 🌟 NEW: Class count for weight calculation
    CLASS_COUNTS = {'0': 503, '1': 116}
    TOTAL_SAMPLES = sum(CLASS_COUNTS.values())

print(f"Using device: {StaticConfig.DEVICE}")


# --- 3. Run Experiment Function (UPDATED) ---

def run_experiment(
    experiment_name: str,
    data_dir: str,
    model_name: str,
    num_epochs: int,
    balance_dataset: bool,
    use_class_weights: bool = False, # NEW parameter
    batch_size: int = 5,
    learning_rate: float = 0.0001,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[str, float]:
    """
    Sets up, trains, and evaluates a complete machine learning pipeline,
    saving all results to an experiment-specific folder.

    Returns:
        tuple[str, float]: (model_path, test_accuracy)
    """

    # --- Setup Output Directory ---
    OUTPUT_DIR = os.path.join("experiments", experiment_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"Data Dir: {data_dir}")
    print(f"Model: {model_name}, Epochs: {num_epochs}, Balanced: {balance_dataset}, Weighted: {use_class_weights}") # Print new setting
    print("="*80)

    # Check if data directory exists before proceeding
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found for experiment {experiment_name}: {data_dir}")

    # --- Data Loading and Splitting ---
    print("\n[Step 1] Loading and splitting data...")
    # NOTE: The balance_dataset (oversample_train) and use_class_weights are often mutually exclusive.
    train_loader, val_loader, test_loader = get_data_loaders(
        root_path=data_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=StaticConfig.RANDOM_SEED,
        oversample_train=balance_dataset
    )

    # --- Class Weight Calculation (NEW LOGIC) ---
    weights_tensor = None
    if use_class_weights:
        # Calculate Inverse Frequency Weights: W_j = N / (N_j * C)
        # A simpler, common form is: W_j = Total Samples / Class_j Samples
        counts = [StaticConfig.CLASS_COUNTS[c] for c in StaticConfig.CLASSES]
        total = StaticConfig.TOTAL_SAMPLES

        # Calculate the weight for each class
        class_weights_list = [total / count for count in counts]

        # Normalize the weights (optional, but often helpful for better comparison across experiments)
        # weights_tensor = torch.tensor(class_weights_list, dtype=torch.float)
        # weights_tensor = weights_tensor / weights_tensor.sum() * len(StaticConfig.CLASSES) # Scales average weight to 1

        weights_tensor = torch.tensor(class_weights_list, dtype=torch.float)

    # --- Model Setup ---
    print(f"\n[Step 2] Setting up model: {model_name}")
    model = get_model(model_name=model_name, num_classes=len(StaticConfig.CLASSES))
    model.to(StaticConfig.DEVICE)

    # Note: We pass the loss CLASS (nn.CrossEntropyLoss) now, not an instance,
    # so the run_training_loop can instantiate it with weights.
    criterion_cls = nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop Execution (UPDATED CALL) ---
    metrics_history, total_time = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_cls=criterion_cls, # Pass the class definition
        optimizer=optimizer,
        device=StaticConfig.DEVICE,
        num_epochs=num_epochs,
        class_weights=weights_tensor # Pass the calculated weights (None if not weighted)
    )
    print(f"\nTotal training duration: {total_time:.2f} minutes.")


    # --- Final Test Set Evaluation ---
    print("\n[Step 3] Final Test Set Evaluation and Saving Outputs...")

    # Create a non-weighted criterion for a standardized test loss (optional, but good practice)
    # The training function uses the weighted one, but we use a standard one here.
    test_criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, true_labels, predictions = validate_one_epoch(
        model,
        test_loader,
        test_criterion, # Use standard criterion for test loss reporting
        StaticConfig.DEVICE,
        return_preds=True
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # ... [Rest of Step 3: Saving Outputs is unchanged] ...

    # Record final test metrics for CSV
    metrics_history.append({
        'epoch': 'Test',
        'train_loss': '',
        'train_acc': '',
        'val_loss': test_loss,
        'val_acc': test_acc
    })

    # --- Saving Outputs ---

    CM_RAW_FILENAME = os.path.join(OUTPUT_DIR, f"{experiment_name}_cm_raw.png")
    CM_NORM_FILENAME = os.path.join(OUTPUT_DIR, f"{experiment_name}_cm_normalized.png")
    ROC_FILENAME = os.path.join(OUTPUT_DIR, f"{experiment_name}_roc_curve.png")
    PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, f"{experiment_name}_predictions.csv")
    METRICS_CSV = os.path.join(OUTPUT_DIR, f"{experiment_name}_metrics.csv")
    MODEL_PATH = os.path.join(OUTPUT_DIR, f"{experiment_name}_model.pth")

    # 1. Generate Confusion Matrices
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, StaticConfig.CLASSES, CM_RAW_FILENAME, title=f"CM Raw ({experiment_name})")
    plot_confusion_matrix(cm, StaticConfig.CLASSES, CM_NORM_FILENAME, normalize=True, title=f"CM Normalized ({experiment_name})")

    # 2. Plot ROC Curve
    plot_roc_curve(model, test_loader, StaticConfig.DEVICE, StaticConfig.CLASSES, ROC_FILENAME)

    # 3. Save Detailed Predictions to CSV
    save_test_predictions_to_csv(model, test_loader, StaticConfig.DEVICE, PREDICTIONS_CSV)

    # 4. Save Metrics to CSV
    save_metrics_to_csv(metrics_history, METRICS_CSV)

    # 5. Save the final model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("\n" + "="*80)
    print(f"EXPERIMENT {experiment_name} COMPLETE. Test Accuracy: {test_acc:.4f}")
    print("="*80)

    return MODEL_PATH, test_acc


# --- 4. Main Execution Block for Running Multiple Experiments (UPDATED) ---

if __name__ == "__main__":

    # Define paths to your different dataset versions
    DATASET_A = '/mnt/data/cases-2/datasets/resample_crop' # e.g., your initial dataset
    DATASET_B = '/mnt/data/cases-2/datasets/resample_shrink' # e.g., a new, differently preprocessed dataset
    DATASET_C = '/mnt/data/cases-2/datasets/no_resample_crop'
    DATASET_D = '/mnt/data/cases-2/datasets/no_resample_shrink'

    ALL_RESULTS = []

    # --- Define Experiments (UPDATED with 'weighted' parameter) ---
    experiments_to_run = [
        {
            "name": "UNet3D_DatasetD_Weighted_20",
            "model": "UNet3D",
            "epochs": 20,
            "balance": False, # Oversampling
            "weighted": True, # NEW: Do not use class weights here
            "data_path": DATASET_D
        }
        # ,
        # {
        #     "name": "R3D18_DatasetC_20",
        #     "model": "R3D18",
        #     "epochs": 20,
        #     "balance": False, # No Oversampling
        #     "weighted": False, # NEW: Use class weights here
        #     "data_path": DATASET_C
        # }
        # ,
        # {
        #     "name": "R3D18_DatasetC_Balanced_20",
        #     "model": "R3D18",
        #     "epochs": 20,
        #     "balance": True,
        #     "weighted": False,
        #     "data_path": DATASET_C
        # },
        # {
        #     "name": "R3D18_DatasetD_Balanced_20",
        #     "model": "R3D18",
        #     "epochs": 20,
        #     "balance": True,
        #     "weighted": False,
        #     "data_path": DATASET_D
        # }
        # Add more experiments comparing Oversampling vs. Class Weighting
    ]

    # --- Run Experiments in Sequence ---
    for exp_config in experiments_to_run:
        try:
            model_path, test_acc = run_experiment(
                experiment_name=exp_config["name"],
                data_dir=exp_config["data_path"],
                model_name=exp_config["model"],
                num_epochs=exp_config["epochs"],
                balance_dataset=exp_config["balance"],
                use_class_weights=exp_config["weighted"], # Pass the new flag
                learning_rate=exp_config.get("lr", 0.0001)
            )
            ALL_RESULTS.append({
                "Experiment": exp_config["name"],
                "Model": exp_config["model"],
                "Balanced": exp_config["balance"],
                "Weighted": exp_config["weighted"], # Add to results
                "Data_Path": exp_config["data_path"],
                "Test_Accuracy": f"{test_acc:.4f}",
                "Model_Path": model_path
            })
        except FileNotFoundError as e:
            print(f"\nFATAL ERROR: Data path missing for experiment {exp_config['name']}. {e}")
            ALL_RESULTS.append({
                "Experiment": exp_config["name"],
                "Model": exp_config["model"],
                "Balanced": exp_config["balance"],
                "Weighted": exp_config["weighted"], # Add to results
                "Data_Path": exp_config["data_path"],
                "Test_Accuracy": "FAILED (Data Not Found)",
                "Model_Path": ""
            })
        except Exception as e:
            print(f"\nFATAL ERROR running experiment {exp_config['name']}: {e}")
            ALL_RESULTS.append({
                "Experiment": exp_config["name"],
                "Model": exp_config["model"],
                "Balanced": exp_config["balance"],
                "Weighted": exp_config["weighted"], # Add to results
                "Data_Path": exp_config["data_path"],
                "Test_Accuracy": "FAILED (Runtime Error)",
                "Model_Path": ""
            })

    # --- Print Summary of All Experiments ---
    print("\n\n" + "*"*80)
    print("GLOBAL EXPERIMENT SUMMARY")
    print("*"*80)
    for result in ALL_RESULTS:
        print(f"[{result['Experiment']}] Model: {result['Model']}, Data: {os.path.basename(result['Data_Path'])}, Balanced: {result['Balanced']}, Weighted: {result['Weighted']}, Acc: {result['Test_Accuracy']}")
    print("*"*80)