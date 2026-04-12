"""
training.py

Implements the per-epoch training and validation loops and the full multi-epoch
training loop for a single cross-validation fold. Responsible for computing
the validation F2-score each epoch and checkpointing the model state that
maximises it.

## Why F2-score as the checkpointing criterion?

Standard accuracy is a poor metric for imbalanced medical datasets. A model
that always predicts "no aneurysm" can achieve 80%+ accuracy while being
clinically useless. F1-score balances precision and recall equally, but that
is also the wrong trade-off for aneurysm detection: a false negative (missed
aneurysm) is far more dangerous than a false positive (unnecessary follow-up
scan). F-beta score generalises F1 by weighting recall β² times more heavily
than precision. With β=2, recall counts four times as much as precision in the
harmonic mean, making the model pay a steep penalty for missed aneurysms.

The best-F2 checkpoint is what the orchestrator uses for final test evaluation,
so the saved model is the one that best avoided false negatives on the
validation set — not merely the last epoch or the one with the lowest loss.

## Why pass criterion_cls (a class) instead of a criterion instance?

``run_training_loop`` receives the loss criterion as a *class* (e.g.
``torch.nn.CrossEntropyLoss``) rather than an instance. This allows the
function to instantiate the criterion internally with the correct ``weight``
tensor for the current fold's device without the caller having to manage
device placement of the weight tensor. If an instance were passed in, the
caller would need to move the weights to the correct device before
constructing it — a coupling that is easy to get wrong. The isinstance guard
at the top of ``run_training_loop`` enforces this contract and gives a clear
error message if an instance is passed by mistake.
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
    """
    Runs one full pass over the training set, updating model weights.

    **Training mode:** ``model.train()`` enables dropout and batch
    normalisation layers to behave in training mode. Specifically, batch
    normalisation uses the statistics of the current mini-batch rather than
    the running statistics accumulated during training, and dropout randomly
    zeroes activations to regularise the network. Forgetting to call
    ``model.train()`` before training (or after a validation pass) is a
    common bug that causes the model to train with frozen batch-norm statistics.

    **Loss accumulation:** Loss is accumulated as
    ``running_loss += loss.item() * inputs.size(0)`` — the per-batch mean
    loss multiplied by the batch size. Dividing the total at the end by
    ``total_samples`` produces the true average loss per sample across the
    epoch, regardless of whether the last batch is smaller than the rest
    (which it often is).

    **Gradient zeroing:** ``optimizer.zero_grad()`` is called at the start
    of each batch, not at the end. Both orderings are valid, but zeroing
    before the forward pass is the conventional PyTorch pattern and avoids
    accidentally carrying stale gradients into the first batch.

    **Tabular (multimodal) support:** If the batch contains a ``"tabular"``
    key (present when ``USE_TABULAR=True`` is set in the experiment config),
    the forward call becomes ``model(inputs, batch["tabular"])``. Otherwise
    it is the standard ``model(inputs)``. The training loop does not need to
    know which mode is active — it detects it from the batch contents, so
    the same function handles both image-only and multimodal experiments.

    Args:
        model: The network to train. Must already be on ``device``. When
            ``USE_TABULAR=True``, this will be a ``MultiModalWrapper`` whose
            forward signature is ``forward(image, tabular)``.
        dataloader: Training DataLoader yielding batches of
            ``{"image": tensor, "label": tensor, "path": str}`` and
            optionally ``"tabular": tensor`` when multimodal fusion is active.
        criterion: Instantiated loss function (e.g. ``CrossEntropyLoss``
            with class weights already applied).
        optimizer: Optimiser holding references to the model parameters.
        device: Target device string (``"cuda"`` or ``"cpu"``).

    Returns:
        Tuple of ``(epoch_loss, epoch_acc)`` — the mean per-sample loss and
        the fraction of correctly classified samples over the full epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for batch in progress_bar:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        if "tabular" in batch:
            outputs = model(inputs, batch["tabular"].to(device))
        else:
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
    Runs one full pass over a validation or test set without updating weights.

    **Evaluation mode:** ``model.eval()`` switches batch normalisation to use
    its accumulated running statistics (not the current batch) and disables
    dropout. This gives deterministic, representative predictions. Forgetting
    this call causes validation metrics to be noisy and not representative of
    true model performance.

    **No-gradient context:** ``torch.no_grad()`` disables the autograd engine
    for the duration of the validation pass. This halves memory usage (no
    intermediate activations are stored for backprop) and speeds up inference.
    It is not a substitute for ``model.eval()`` — both are required.

    **F2-score computation:** All predictions and labels are collected across
    batches into flat lists, and ``fbeta_score`` is called once on the full
    epoch at the end. Computing F2 per-batch and averaging would give an
    incorrect result because F2 is not linear — it depends on the global
    confusion matrix, not an average of per-batch confusion matrices.
    ``zero_division=0`` returns 0 rather than raising a warning when the
    model predicts no positive cases (which can happen early in training).

    **Detailed results:** When ``return_details=True``, the function records
    per-sample ``(filename, true_label, pred_label)`` dicts for the final
    test-set evaluation pass. This data is used by the orchestrator to save
    per-fold prediction CSVs and compute aggregate confusion matrices. The
    flag is ``False`` by default so the per-epoch validation loop during
    training does not accumulate this overhead unnecessarily.

    **Tabular (multimodal) support:** Mirrors the behaviour in
    ``train_one_epoch`` — if the batch contains a ``"tabular"`` key, the
    forward call is ``model(inputs, batch["tabular"])``; otherwise it is
    ``model(inputs)``. The same DataLoader that was used for training will
    include the tabular key for evaluation if ``USE_TABULAR=True``.

    Args:
        model: The network to evaluate. Must already be on ``device``. When
            ``USE_TABULAR=True``, this will be a ``MultiModalWrapper``.
        dataloader: Validation or test DataLoader yielding batches of
            ``{"image": tensor, "label": tensor, "path": str}`` and
            optionally ``"tabular": tensor`` when multimodal fusion is active.
        criterion: Instantiated loss function (same criterion used in training,
            so val loss is directly comparable to train loss).
        device: Target device string.
        return_details: If ``True``, return per-sample prediction dicts as a
            fourth element of the return tuple.

    Returns:
        Always returns ``(epoch_loss, epoch_acc, val_f2)``. When
        ``return_details=True``, also returns a list of per-sample dicts as
        a fourth element, each containing ``"filename"``, ``"true_label"``,
        and ``"pred_label"``.
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
    patience: int = 0,
) -> Tuple[List[Dict[str, Any]], float, Optional[Dict[str, Any]]]:
    """
    Trains a model for ``num_epochs`` epochs and returns the best-F2 checkpoint.

    Each epoch calls ``train_one_epoch`` followed by ``validate_one_epoch``.
    After every validation pass the current val F2 is compared to the best
    seen so far; if it improves, a deep copy of the model's state dict is saved
    as the new checkpoint.

    **Why deep copy the state dict?**
    ``model.state_dict()`` returns a dict of references to the model's
    parameter tensors. Without ``copy.deepcopy``, the saved checkpoint dict
    would point to the same underlying tensors as the live model, meaning
    subsequent training epochs would silently overwrite the "saved" weights.
    ``copy.deepcopy`` creates independent copies so the checkpoint remains
    frozen at the moment it was taken.

    **Checkpoint initialisation:** ``best_val_f2`` starts at 0.0. A model
    is only checkpointed when it produces F2 > 0, i.e. it correctly identifies
    at least one positive case. If no epoch achieves this (the model collapses
    to always predicting negative), ``best_model_checkpoint`` remains ``None``
    and a warning is printed. The orchestrator handles the ``None`` case.

    **Early stopping:** When ``patience > 0``, training halts if val F2 has
    not improved for ``patience`` consecutive epochs. This prevents wasting
    compute on epochs that would only overfit. When ``patience=0`` (default),
    all ``num_epochs`` epochs are always run. Early stopping does not affect
    which checkpoint is saved — the best epoch is still the one with the
    highest F2, regardless of when training stops.

    **Criterion instantiation:** ``criterion_cls`` (e.g.
    ``torch.nn.CrossEntropyLoss``) is instantiated inside this function with
    the class weights tensor already placed on ``device``. See the module
    docstring for why a class is passed rather than an instance.

    Args:
        model: Network to train. Must already be on ``device``.
        train_loader: DataLoader for the training fold.
        val_loader: DataLoader for the validation fold.
        criterion_cls: Loss function *class* (not an instance). Must accept
            a ``weight`` keyword argument if ``class_weights`` is provided.
        optimizer: Configured optimiser (e.g. ``AdamW``), already linked to
            ``model.parameters()``.
        device: Target device string (``"cuda"`` or ``"cpu"``).
        num_epochs: Maximum number of epochs to train.
        class_weights: Optional float tensor of shape ``[num_classes]``
            containing per-class loss weights. Passed to ``criterion_cls``
            as ``weight=class_weights.to(device)``. ``None`` disables
            weighted loss.
        patience: Early stopping patience in epochs. ``0`` disables early
            stopping.

    Returns:
        A tuple of:
          - ``metrics_history``: List of per-epoch dicts, each containing
            ``epoch``, ``train_loss``, ``train_acc``, ``val_loss``,
            ``val_acc``, ``val_f2``.
          - ``total_training_time_minutes``: Wall-clock training duration
            in minutes.
          - ``best_model_checkpoint``: Dict with ``state_dict``,
            ``best_epoch``, ``best_val_f2`` for the epoch that achieved the
            highest val F2. ``None`` if no epoch produced F2 > 0.

    Raises:
        TypeError: If ``criterion_cls`` is an instance rather than a class.
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
    epochs_without_improvement = 0
    start_time = time.time()

    if patience > 0:
        print(f"\nStarting training (early stopping patience={patience})...")
    else:
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
            epochs_without_improvement = 0
            print(
                f"  -> New best model: epoch {epoch + 1}, val F2 = {best_val_f2:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if patience > 0:
                print(
                    f"  -> No improvement ({epochs_without_improvement}/{patience})"
                )
                if epochs_without_improvement >= patience:
                    print(
                        f"  -> Early stopping triggered at epoch {epoch + 1}."
                    )
                    metrics_history.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f2": val_f2,
                    })
                    break

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
