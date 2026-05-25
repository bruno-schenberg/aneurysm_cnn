"""
training.py

Implements the per-epoch training and validation loops and the full multi-epoch
training loop for a single cross-validation fold.

- We checkpoint on ROC-AUC because it is threshold-independent and more stable
  than F2 on small, imbalanced validation sets. F2 (which weights recall over
  precision) is still computed each epoch and recorded in the metrics history for
  reference, but AUC drives the model selection decision.
- ``run_training_loop`` receives the loss criterion as a class rather than an
  instance so it can instantiate it internally with the correct weight tensor
  for the current fold's device.
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from sklearn.metrics import fbeta_score, roc_auc_score


# ----------------------------------------------------
# 1. Per-Epoch Training
# ----------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Runs one full pass over the training set, updating model weights.

    Args:
        model: The network to train.
        dataloader: Training DataLoader yielding batches of
            ``{"image": tensor, "label": tensor, "path": str}`` and
            optionally ``"tabular": tensor``.
        criterion: Instantiated loss function.
        optimizer: Optimiser holding references to the model parameters.
        device: Target device string (``"cuda"`` or ``"cpu"``).
        grad_accum_steps: Number of mini-batches to accumulate gradients over.
        use_amp: Use mixed precision training.

    Returns:
        Tuple of ``(epoch_loss, epoch_acc)``.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(dataloader)

    _device_type = "cuda" if "cuda" in str(device) else "cpu"
    _amp_enabled = use_amp and _device_type == "cuda"

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        with torch.amp.autocast(device_type=_device_type, dtype=torch.bfloat16, enabled=_amp_enabled):
            if "tabular" in batch:
                outputs = model(inputs, batch["tabular"].to(device))
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        (loss / grad_accum_steps).backward()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == num_batches
        if is_accum_step or is_last_batch:
            optimizer.step()
            optimizer.zero_grad()

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
    use_amp: bool = True,
) -> Union[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float, List[Dict[str, Any]]],
]:
    """
    Runs one full pass over a validation or test set without updating weights.

    Args:
        model: The network to evaluate.
        dataloader: Validation or test DataLoader.
        criterion: Instantiated loss function.
        device: Target device string.
        return_details: If ``True``, return per-sample prediction dicts as a
            fifth element of the return tuple.
        use_amp: Use mixed precision.

    Returns:
        ``(epoch_loss, epoch_acc, val_f2, val_auc)``. When ``return_details=True``,
        also returns a list of per-sample dicts as the fifth element.
        ``val_auc`` is set to ``0.0`` when the validation set contains only one
        class (AUC is undefined in that case).
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    detailed_results: List[Dict[str, Any]] = []

    _device_type = "cuda" if "cuda" in str(device) else "cpu"
    _amp_enabled = use_amp and _device_type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            with torch.amp.autocast(device_type=_device_type, dtype=torch.bfloat16, enabled=_amp_enabled):
                if "tabular" in batch:
                    outputs = model(inputs, batch["tabular"].to(device))
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            # Collect P(class=1) for threshold-independent AUC computation
            probs = torch.softmax(outputs.float(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())

            if return_details:
                paths = batch["path"]
                for i in range(len(paths)):
                    detailed_results.append({
                        "filename": paths[i],
                        "true_label": labels[i].item(),
                        "pred_label": preds[i].item(),
                    })

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    val_f2 = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Only one class present in the validation set — AUC is undefined
        val_auc = 0.0

    if return_details:
        return epoch_loss, epoch_acc, val_f2, val_auc, detailed_results
    return epoch_loss, epoch_acc, val_f2, val_auc


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
    patience: int = 0,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
) -> Tuple[List[Dict[str, Any]], float, Optional[Dict[str, Any]]]:
    """
    Trains a model for ``num_epochs`` epochs and returns the best-AUC checkpoint.

    Checkpointing is driven by ``val_auc`` (ROC-AUC), which is more stable than
    F2 on small, imbalanced validation sets because it is threshold-independent.
    ``val_f2`` is still tracked in ``metrics_history`` for reporting.

    Args:
        model: Network to train.
        train_loader: DataLoader for the training fold.
        val_loader: DataLoader for the validation fold.
        criterion_cls: Loss function class (not an instance).
        optimizer: Configured optimiser.
        device: Target device string.
        num_epochs: Maximum number of epochs to train.
        class_weights: Optional float tensor of shape ``[num_classes]``.
        patience: Early stopping patience in epochs. ``0`` disables early stopping.
        grad_accum_steps: Number of mini-batches to accumulate gradients over.
        use_amp: Use mixed precision.

    Returns:
        A tuple of:
          - ``metrics_history``: List of per-epoch dicts (includes ``val_f2`` and ``val_auc``).
          - ``total_training_time_minutes``: Wall-clock training duration in minutes.
          - ``best_model_checkpoint``: Dict with ``state_dict``, ``best_epoch``,
            ``best_val_auc``, and ``best_val_f2``.
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
    best_val_auc = 0.0
    best_model_checkpoint: Optional[Dict[str, Any]] = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_accum_steps=grad_accum_steps, use_amp=use_amp,
        )
        val_loss, val_acc, val_f2, val_auc = validate_one_epoch(
            model, val_loader, criterion, device, use_amp=use_amp,
        )

        epoch_duration = time.time() - epoch_start_time

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_checkpoint = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "best_epoch": epoch + 1,
                "best_val_auc": val_auc,
                "best_val_f2": val_f2,
            }
            epochs_without_improvement = 0
            print(
                f"  -> [Epoch {epoch + 1:03d} | {epoch_duration:.1f}s] "
                f"New best model: val AUC = {best_val_auc:.4f} (F2 = {val_f2:.4f})"
            )
        else:
            epochs_without_improvement += 1
            if patience > 0:
                if epochs_without_improvement >= patience:
                    metrics_history.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f2": val_f2,
                        "val_auc": val_auc,
                        "duration_sec": epoch_duration,
                    })
                    break

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f2": val_f2,
            "val_auc": val_auc,
            "duration_sec": epoch_duration,
        })

    total_training_time = (time.time() - start_time) / 60

    if best_model_checkpoint is None:
        print(
            "Warning: No epoch produced a positive AUC score. "
            "No best-model checkpoint was saved."
        )

    return metrics_history, total_training_time, best_model_checkpoint
