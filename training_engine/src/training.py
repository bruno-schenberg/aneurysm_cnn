import torch
from tqdm import tqdm
import time # Added for run_training_loop
import copy # Added to copy model state
from typing import Tuple, List, Dict, Union, Any, Optional, Type # Added Optional, Type for run_training_loop

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for batch in progress_bar:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        progress_bar.set_postfix(
            loss=loss.item(), acc=correct_predictions.double().item() / total_samples
        )
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    return_details: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, List[Dict[str, Any]]]]:
    """Performs one epoch of validation, optionally returning detailed results."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    detailed_results = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
        # The dataloader yields a dictionary batch
        for batch in progress_bar:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            if return_details:
                paths = batch["path"]
                for i in range(len(paths)):
                    detailed_results.append({
                        "filename": paths[i],
                        "true_label": labels[i].item(),
                        "pred_label": preds[i].item(),
                    })
            progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions.double().item()/total_samples)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    if return_details:
        return epoch_loss, epoch_acc.item(), detailed_results
    else:
        return epoch_loss, epoch_acc.item()

def run_training_loop(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader, criterion_cls: Type[torch.nn.Module],
                      optimizer: torch.optim.Optimizer, device: str,
                      num_epochs: int,
                      class_weights: Optional[torch.Tensor] = None
                      ) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """
    Runs the full training and validation loop for the specified number of epochs.
    Tracks the model with the best validation loss and returns its state dictionary.

    Returns:
        tuple[List[Dict[str, Any]], float]: (metrics_history, total_training_time_minutes)
    """

    # Ensure criterion_cls is a class type, not an instance.
    if not isinstance(criterion_cls, type):
        raise TypeError("criterion_cls must be the class type (e.g., torch.nn.CrossEntropyLoss), not an instance.")

    # Initialize the criterion with optional class weights
    if class_weights is not None:
        criterion = criterion_cls(weight=class_weights.to(device))
    else:
        criterion = criterion_cls()

    metrics_history = []
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_model_state = None
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Use the instantiated criterion for training and validation
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # NOTE: It's common practice to use the weighted criterion for validation loss reporting too,
        # but if you need an unweighted validation loss, you'd create an unweighted criterion here.
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}  | Val Acc: {val_acc:.4f}")

        # Check if this is the best model so far based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model found with validation loss: {best_val_loss:.4f}")

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

    return metrics_history, total_training_time, best_model_state