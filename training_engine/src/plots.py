import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import itertools
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import csv
from typing import List, Dict, Any
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, fbeta_score

# --- Plotting and Saving Functions ---

def plot_confusion_matrix(cm, classes, filename, normalize=False, title='Confusion Matrix'):
    """Plots and saves the confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def save_predictions_from_detailed_results(detailed_results: List[Dict[str, Any]], filename: str):
    """
    Saves detailed predictions to a CSV from a pre-computed list of results.
    This avoids re-running inference.
    """
    results = []
    for item in detailed_results:
        # The 'filename' key from detailed_results is the base name
        results.append({
            'file_name': item['filename'],
            'true_label': item['true_label'],
            'prediction': item['pred_label']
        })

    csv_headers = ['file_name', 'true_label', 'prediction']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"Test set predictions saved to {filename}")

def plot_roc_curve(model, dataloader, device, classes, filename):
    """Computes and plots the ROC curve."""
    model.eval()
    all_labels = []
    all_probs = []

    progress_bar = tqdm.tqdm(dataloader, desc="Calculating ROC", unit="batch")
    with torch.no_grad():
        # Dataloader now yields a dictionary batch
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
        print("Warning: Only one class found in test set labels. Cannot generate ROC curve.")
        return

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    print(f"ROC curve saved to {filename}")

def save_metrics_to_csv(
    metrics_history: List[Dict[str, Any]],
    filename: str,
    detailed_results: List[Dict[str, Any]] = None
):
    """
    Saves the training/validation metrics history to a CSV file.
    If detailed_results are provided, it also calculates and appends
    final classification metrics (Precision, Recall, F2).
    """
    csv_headers = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(metrics_history)
        
        if detailed_results:
            final_metrics = calculate_classification_metrics(detailed_results)
            writer.writerow({}) # Add a blank line for separation
            writer.writerow({'epoch': 'Metric', 'train_loss': 'Value'}) # Use epoch/train_loss as headers
            writer.writerow({'epoch': 'Precision', 'train_loss': f"{final_metrics['Precision']:.4f}"})
            writer.writerow({'epoch': 'Recall', 'train_loss': f"{final_metrics['Recall']:.4f}"})
            writer.writerow({'epoch': 'F2-Score', 'train_loss': f"{final_metrics['F2-Score']:.4f}"})
    print(f"Metrics saved to {filename}")

def calculate_classification_metrics(detailed_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates Precision, Recall, and F2-Score from a list of detailed predictions.

    Args:
        detailed_results (List[Dict[str, Any]]): A list of prediction dictionaries.

    Returns:
        Dict[str, float]: A dictionary containing 'Precision', 'Recall', and 'F2-Score'.
                          Returns a dictionary of NaNs if calculation fails.
    """
    try:
        df = pd.DataFrame(detailed_results)
        y_true = df['true_label'].astype(int)
        y_pred = df['pred_label'].astype(int)

        if len(y_true.unique()) < 2:
            print("Warning: Only one class present in true labels. Metrics will be 0.")
            return {'Precision': 0.0, 'Recall': 0.0, 'F2-Score': 0.0}

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f2_score = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)

        return {
            'Precision': precision,
            'Recall': recall,
            'F2-Score': f2_score
        }
    except Exception as e:
        print(f"An error occurred during metric calculation: {e}")
        return {'Precision': float('nan'), 'Recall': float('nan'), 'F2-Score': float('nan')}

def evaluate_models(all_fold_predictions: Dict[str, List[Dict[str, Any]]], output_csv_path: str):
    """
    Loops through a dictionary of predictions, calculates Precision, Recall,
    and F2-Score for each fold, prints the results, and saves them to a CSV.

    Args:
        all_fold_predictions (Dict[str, List[Dict[str, Any]]]): A dictionary where keys
            are model/fold names and values are the `detailed_results` list for that fold.
        output_csv_path (str): The path to save the summary CSV file.
    """
    print(f"\nEvaluating predictions for {len(all_fold_predictions)} folds...")

    # List to store the results for all models
    results = []

    # Loop through all files in the specified folder
    for model_name, fold_data in all_fold_predictions.items():
        if fold_data and 'metrics' in fold_data:
            print(f"--- Aggregating results for **{model_name}** ---")
            # Combine all metrics for this fold into a single dictionary
            fold_results = {
                'Model Name': model_name,
                **fold_data['metrics'], # Precision, Recall, F2-Score
                'Eval Accuracy': fold_data['eval_acc'],
                'Eval Loss': fold_data['eval_loss']
            }
            results.append(fold_results)
            print(f"  Eval Acc: {fold_results['Eval Accuracy']:.4f}, Eval Loss: {fold_results['Eval Loss']:.4f}, F2-Score: {fold_results['F2-Score']:.4f}")

    # Create and display final summary table
    if results:
        results_df = pd.DataFrame(results)

        # Calculate Averages and Std Deviations
        avg_metrics = {
            'Model Name': 'Average',
            'Precision': results_df['Precision'].mean(),
            'Recall': results_df['Recall'].mean(),
            'F2-Score': results_df['F2-Score'].mean(),
            'Eval Accuracy': results_df['Eval Accuracy'].mean(),
            'Eval Loss': results_df['Eval Loss'].mean()
        }
        std_metrics = {
            'Model Name': 'Std. Dev.',
            'Precision': results_df['Precision'].std(),
            'Recall': results_df['Recall'].std(),
            'F2-Score': results_df['F2-Score'].std(),
            'Eval Accuracy': results_df['Eval Accuracy'].std(),
            'Eval Loss': results_df['Eval Loss'].std()
        }
        results_df = pd.concat([results_df, pd.DataFrame([avg_metrics, std_metrics])], ignore_index=True)

        # Sort by F2-Score to easily see the best performing model based on your priority
        print("\n" + "="*50)
        print("📊 FINAL MODEL EVALUATION SUMMARY")
        print("="*50)
        print(results_df.to_string(index=False, float_format="%.4f"))
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nEvaluation summary saved to: {output_csv_path}")
    else:
        print("\nNo valid fold data was processed.")

def save_fold_artifacts(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    best_model_state: Dict[str, Any],
    detailed_results: List[Dict[str, Any]],
    metrics_history: List[Dict[str, Any]],
    experiment_name: str,
    fold: int,
    fold_output_dir: str,
    config: Dict[str, Any]
):
    """
    Generates and saves all artifacts for a single fold after training and evaluation.
    This includes plots, metrics, predictions, and model states.
    """
    print(f"\n[Fold {fold}] Generating and saving evaluation artifacts...")

    # Define paths for all output files for this fold
    CM_RAW_FILENAME = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_cm_raw.png")
    CM_NORM_FILENAME = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_cm_normalized.png")
    ROC_FILENAME = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_roc_curve.png")
    PREDICTIONS_CSV = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_predictions.csv")
    METRICS_CSV = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_metrics.csv")
    LAST_MODEL_PATH = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_model_last.pth")
    BEST_MODEL_PATH = os.path.join(fold_output_dir, f"{experiment_name}_fold{fold}_model_best.pth")

    # --- 1. Generate artifacts from pre-computed 'detailed_results' (Efficient) ---
    true_labels = [r['true_label'] for r in detailed_results]
    predictions = [r['pred_label'] for r in detailed_results]
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrices
    plot_confusion_matrix(cm, config['CLASSES'], CM_RAW_FILENAME, title=f"CM Raw (Fold {fold})")
    plot_confusion_matrix(cm, config['CLASSES'], CM_NORM_FILENAME, normalize=True, title=f"CM Norm (Fold {fold})")

    # Save predictions CSV without re-running inference
    save_predictions_from_detailed_results(detailed_results, PREDICTIONS_CSV)

    # --- 2. Generate artifacts that require re-running inference ---
    # ROC curve requires probabilities, not just final labels, so inference is needed here.
    plot_roc_curve(model, eval_loader, config['DEVICE'], config['CLASSES'], ROC_FILENAME)

    # --- 3. Save training history and final metrics to a single CSV ---
    # The function now handles both epoch history and final metrics calculation.
    save_metrics_to_csv(
        metrics_history=metrics_history,
        filename=METRICS_CSV,
        detailed_results=detailed_results
    )

    # --- 4. Save model states ---
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"Last model for fold {fold} saved to {LAST_MODEL_PATH}")

    if best_model_state:
        torch.save(best_model_state, BEST_MODEL_PATH)
        print(f"Best model for fold {fold} saved to {BEST_MODEL_PATH}")
