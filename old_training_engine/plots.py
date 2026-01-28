import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import itertools
import os
from sklearn.metrics import roc_curve, auc
import csv
from typing import List, Dict, Any

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

def save_test_predictions_to_csv(model, dataloader, device, filename):
    """Saves detailed predictions (filename, true label, prediction) to a CSV."""
    model.eval()
    results = []

    progress_bar = tqdm.tqdm(dataloader, desc="Saving Predictions", unit="batch")
    with torch.no_grad():
        # Dataloader returns (inputs, labels, file_paths)
        for inputs, labels, file_paths in progress_bar:

            # Note: We need to handle the case where the dataset returns indices
            # and we need to map those back to the correct file paths if we weren't
            # using the full file_paths list from the base NiftiFolder, but for
            # the current setup where Subset indices map to the NiftiFolder's sample list,
            # this works because the DataLoader receives the list of paths directly.

            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for path, true_label, prediction in zip(file_paths, labels.cpu().numpy(), preds.cpu().numpy()):
                file_name = os.path.basename(path)
                results.append({
                    'file_name': file_name,
                    'true_label': true_label.item(),
                    'prediction': prediction.item()
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
        # Dataloader returns (inputs, labels, file_paths), use _ for file_paths
        for inputs, labels, _ in progress_bar:
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


def save_metrics_to_csv(metrics_history: List[Dict[str, Any]], filename: str):
    """
    Saves the training/validation metrics history to a CSV file.
    """
    csv_headers = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(metrics_history)

    print(f"Metrics saved to {filename}")