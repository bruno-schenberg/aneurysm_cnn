# @title
import torch
from tqdm import tqdm
import time
from typing import Tuple, List, Any, Dict, Union, Optional, Type # Import Optional and Type

# --- Existing Core Training and Validation Logic (No Changes Needed Here) ---

def train_one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                    criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str) -> Tuple[float, float]:
    """
    Performs one epoch of training.

    Returns:
        tuple[float, float]: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, labels, _ in progress_bar:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions.double().item()/total_samples)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def validate_one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                       criterion: torch.nn.Module, device: str,
                       return_preds: bool = False) -> Union[Tuple[float, float], Tuple[float, float, List[Any], List[Any]]]:
    """
    Performs one epoch of validation or testing.

    Returns:
        (loss, accuracy) or (loss, accuracy, true_labels, predictions) if return_preds is True.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc="Validation" if not return_preds else "Testing", unit="batch")
    with torch.no_grad():
        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            if return_preds:
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions.double().item()/total_samples)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    if return_preds:
        return epoch_loss, epoch_acc.item(), all_labels, all_preds
    else:
        return epoch_loss, epoch_acc.item()

# ----------------------------------------------------
# Updated Training Loop Helper Function
# ----------------------------------------------------

def run_training_loop(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader, criterion_cls: Type[torch.nn.Module], # Changed criterion to criterion_cls
                      optimizer: torch.optim.Optimizer, device: str,
                      num_epochs: int,
                      class_weights: Optional[torch.Tensor] = None # New parameter
                      ) -> Tuple[List[Dict[str, Any]], float]:
    """
    Runs the full training and validation loop for the specified number of epochs.
    Allows passing class weights for cost-sensitive learning.

    Returns:
        tuple[List[Dict[str, Any]], float]: (metrics_history, total_training_time_minutes)
    """

    # 🌟 NEW LOGIC: Initialize the criterion with optional class weights
    if class_weights is not None:
        if isinstance(criterion_cls, type): # Check if it's a class definition
            # Ensure weights are on the correct device if provided
            weighted_criterion = criterion_cls(weight=class_weights.to(device))
            print(f"Using class weights in loss function: {class_weights.tolist()}")
        else:
            raise TypeError("criterion_cls must be the class type (e.g., torch.nn.CrossEntropyLoss), not an instance.")
    else:
        # Instantiate without weights if it's a class type
        if isinstance(criterion_cls, type):
            weighted_criterion = criterion_cls()
        else:
            # If an already instantiated criterion was passed, use it (though passing the class is preferred)
            weighted_criterion = criterion_cls
            print("Warning: Passed an instantiated criterion. Class weighting option will be ignored unless it was already applied.")


    metrics_history = []
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Use the weighted_criterion for training and validation
        train_loss, train_acc = train_one_epoch(model, train_loader, weighted_criterion, optimizer, device)

        # NOTE: It's common practice to use the weighted criterion for validation loss reporting too,
        # but if you need an unweighted validation loss, you'd create an unweighted criterion here.
        val_loss, val_acc = validate_one_epoch(model, val_loader, weighted_criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}  | Val Acc: {val_acc:.4f}")

        # Record metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    end_time = time.time()
    total_training_time = (end_time - start_time) / 60
    print(f"\nTraining completed in {total_training_time:.2f} minutes.")

    return metrics_history, total_training_time