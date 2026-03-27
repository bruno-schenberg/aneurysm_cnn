"""
training.py

Implements the per-epoch training and validation loops and the full multi-epoch
training loop for a single cross-validation fold. Responsible for computing
validation F2-score each epoch and checkpointing the model state that maximises it.
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from sklearn.metrics import fbeta_score
from tqdm import tqdm


# ----------------------------------------------------
# 1. Per-Epoch Training
# ----------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Runs one epoch of gradient-descent training. Returns (epoch_loss, epoch_acc)."""
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


# ----------------------------------------------------
# 2. Per-Epoch Validation
# ----------------------------------------------------


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    return_details: bool = False,
) -> Union[
    Tuple[float, float, float],
    Tuple[float, float, float, List[Dict[str, Any]]],
]:
    """
    Runs one epoch of validation. Always returns (epoch_loss, epoch_acc, val_f2).
    When return_details=True, also returns per-sample prediction dicts as a fourth element.

    val_f2 is the F2-score (β=2, recall-weighted) on the positive class, computed
    across all batches. This is the checkpointing criterion used by run_training_loop.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    detailed_results: List[Dict[str, Any]] = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
        for batch in progress_bar:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if return_details:
                paths = batch["path"]
                for i in range(len(paths)):
                    detailed_results.append({
                        "filename": paths[i],
                        "true_label": labels[i].item(),
                        "pred_label": preds[i].item(),
                    })

            progress_bar.set_postfix(
                loss=loss.item(),
                acc=correct_predictions.double().item() / total_samples,
            )

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    val_f2 = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0)

    if return_details:
        return epoch_loss, epoch_acc, val_f2, detailed_results
    return epoch_loss, epoch_acc, val_f2


# ----------------------------------------------------
# 3. Full Training Loop
# ----------------------------------------------------


def run_training_loop(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion_cls: Type[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict[str, Any]], float, Optional[Dict[str, Any]]]:
    """
    Runs the full training and validation loop for the specified number of epochs.
    Checkpoints the model state at the epoch with the highest validation F2-score.

    Returns:
        metrics_history: List of per-epoch dicts (epoch, train_loss, train_acc,
                         val_loss, val_acc, val_f2).
        total_training_time_minutes: Wall-clock training duration in minutes.
        best_model_checkpoint: Dict with 'state_dict', 'best_epoch', 'best_val_f2',
                               or None if no epoch produced a positive F2-score.
    """
    if not isinstance(criterion_cls, type):
        raise TypeError(
            "criterion_cls must be the class type (e.g., torch.nn.CrossEntropyLoss), not an instance."
        )

    criterion = (
        criterion_cls(weight=class_weights.to(device))
        if class_weights is not None
        else criterion_cls()
    )

    metrics_history: List[Dict[str, Any]] = []
    best_val_f2 = 0.0
    best_model_checkpoint: Optional[Dict[str, Any]] = None
    start_time = time.time()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f2 = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )
        print(
            f"Epoch {epoch + 1} | Val Loss:   {val_loss:.4f}  | "
            f"Val Acc: {val_acc:.4f} | Val F2: {val_f2:.4f}"
        )

        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            best_model_checkpoint = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "best_epoch": epoch + 1,
                "best_val_f2": val_f2,
            }
            print(
                f"  -> New best model: epoch {epoch + 1}, val F2 = {best_val_f2:.4f}"
            )

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f2": val_f2,
        })

    total_training_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_training_time:.2f} minutes.")

    if best_model_checkpoint is None:
        print(
            "Warning: No epoch produced a positive F2-score. "
            "No best-model checkpoint was saved."
        )

    return metrics_history, total_training_time, best_model_checkpoint
