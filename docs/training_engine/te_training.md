## training.py

`training_engine/src/training.py` implements the per-epoch training loop, the per-epoch validation loop, and the full multi-epoch training orchestrator for a single cross-validation fold. It has no direct file I/O — all checkpoint state is returned in-memory as Python dicts. The module docstring explains the core design choices: ROC-AUC drives checkpointing, and the loss criterion is received as a class (not an instance) so the correct device can be determined before instantiation.

**External dependencies:** `torch`, `sklearn.metrics` (`fbeta_score`, `roc_auc_score`), `copy`, `time`.

**Experiment config coupling:** This module is intentionally decoupled from any config file format. All parameters (epochs, patience, grad accumulation, AMP, class weights, etc.) are received as explicit Python arguments. The caller (e.g., `run_experiment` in `experiment.py`) is responsible for reading config keys and forwarding them. There are no direct references to config key strings inside this file.

---

### Functions

---

#### `train_one_epoch`

**What it does:** Runs a single full pass over the training DataLoader, accumulating gradients across mini-batches and updating model weights, returning the epoch-average loss and accuracy.

**Design rationale:**
- **Mixed precision (AMP) with `bfloat16`:** `bfloat16` has a wider dynamic range than `float16` (same exponent bits as `float32`) and does not require a `GradScaler`, simplifying the training loop. AMP is only enabled when the device is CUDA; it is silently skipped on CPU where `bfloat16` autocast is less well-supported.
- **Gradient accumulation:** Allows an effective batch size larger than what fits in GPU memory by summing gradients over `grad_accum_steps` mini-batches before calling `optimizer.step()`. The loss is divided by `grad_accum_steps` before `.backward()` so the effective gradient magnitude is equivalent to a single forward pass over the concatenated batch.
- **`optimizer.zero_grad()` before the loop (not inside):** Gradients are zeroed once at the start and again only after each optimizer step, avoiding unnecessary kernel launches on every batch.
- **Tabular branch:** The forward call is guarded by `"tabular" in batch`, making the function work transparently for both unimodal and multimodal models without separate code paths in the caller.

**Parameters:**
- `model` (`torch.nn.Module`): The network to train; set to `model.train()` internally.
- `dataloader` (`torch.utils.data.DataLoader`): Training DataLoader yielding dicts with keys `"image"` (`Tensor`), `"label"` (`Tensor`), `"path"` (`str`), and optionally `"tabular"` (`Tensor`).
- `criterion` (`torch.nn.Module`): Instantiated loss function (e.g., `CrossEntropyLoss`).
- `optimizer` (`torch.optim.Optimizer`): Optimiser holding references to the model parameters.
- `device` (`str`): Target device string (`"cuda"` or `"cpu"`).
- `grad_accum_steps` (`int`, default `1`): Number of mini-batches to accumulate gradients over before calling `optimizer.step()`.
- `use_amp` (`bool`, default `True`): Enable mixed-precision autocast; automatically disabled if `device` is CPU.

**Returns:** `Tuple[float, float]` — `(epoch_loss, epoch_acc)`.
- `epoch_loss`: Running loss summed over samples, divided by total samples (sample-weighted average, not batch-weighted).
- `epoch_acc`: Fraction of correctly classified samples.

**Side effects:** Modifies model parameters in-place via `optimizer.step()`. No logging, no file I/O, no metric tracking beyond the return values.

---

#### `validate_one_epoch`

**What it does:** Runs a single full pass over a validation or test DataLoader under `torch.no_grad()`, computing loss, accuracy, F2 score, and ROC-AUC, with an optional per-sample detail list.

**Design rationale:**
- **`torch.no_grad()`:** Disables gradient computation and autograd graph construction, reducing memory usage and speeding up inference.
- **AMP during validation:** Autocast is kept active during validation (when the device supports it) to ensure logits are computed under the same numerical conditions as training, avoiding potential distribution shift from dtype differences.
- **F2 score:** Beta=2 weights recall twice as heavily as precision, reflecting the clinical asymmetry where missing an aneurysm (false negative) is more costly than a false alarm (false positive).
- **ROC-AUC over F2 for checkpointing:** AUC is threshold-independent, making it more stable on small or imbalanced validation sets where the optimal classification threshold may shift between epochs. F2 is still computed and returned for reporting.
- **`torch.softmax(...).float()`:** The `.float()` cast ensures probability computation happens in full precision even if outputs are in `bfloat16` from AMP, preventing AUC instability from low-precision softmax.
- **AUC fallback to `0.0`:** `roc_auc_score` raises `ValueError` when only one class is present in the validation set. This is caught and replaced with `0.0`, which prevents a crash during early folds or extreme stratification failure, while ensuring the fallback epoch never wins the checkpoint race (any real AUC > 0.0).
- **`return_details` flag:** Per-sample filename/label/prediction dicts are only collected when explicitly requested, avoiding the cost of building and returning a potentially large list on every validation epoch during training.

**Parameters:**
- `model` (`torch.nn.Module`): The network to evaluate; set to `model.eval()` internally.
- `dataloader` (`torch.utils.data.DataLoader`): Validation or test DataLoader; same dict schema as the training loader.
- `criterion` (`torch.nn.Module`): Instantiated loss function.
- `device` (`str`): Target device string.
- `return_details` (`bool`, default `False`): If `True`, collect and return per-sample prediction dicts as the fifth element of the return tuple.
- `use_amp` (`bool`, default `True`): Enable mixed-precision autocast during inference.

**Returns:**
- When `return_details=False`: `Tuple[float, float, float, float]` — `(epoch_loss, epoch_acc, val_f2, val_auc)`.
- When `return_details=True`: `Tuple[float, float, float, float, List[Dict[str, Any]]]` — same four scalars plus a list of per-sample dicts, each with keys `"filename"` (`str`), `"true_label"` (`int`), `"pred_label"` (`int`).

**Side effects:** None — no weight updates, no file I/O, no logging. The `model.eval()` call modifies the model's mode (affecting dropout and batch norm behaviour) but does not change parameters.

---

#### `run_training_loop`

**What it does:** Orchestrates the full multi-epoch training run for one fold: instantiates the loss criterion on the correct device, calls `train_one_epoch` and `validate_one_epoch` each epoch, tracks the best-AUC checkpoint in memory, and optionally triggers early stopping.

**Design rationale:**
- **Criterion received as a class, not an instance:** The class is instantiated inside the function so `class_weights.to(device)` is called after the device is known, avoiding a bug where weights are on CPU when the model is on GPU.
- **Checkpointing on `val_auc`:** ROC-AUC is threshold-independent and more stable than F2 on small, imbalanced validation folds. The best checkpoint is stored as an in-memory dict (not written to disk here) so the caller decides when and whether to serialise it.
- **`copy.deepcopy(model.state_dict())`:** A deep copy is mandatory because `state_dict()` returns a reference to the live tensors; without `deepcopy`, the checkpoint would be silently overwritten by subsequent weight updates.
- **Early stopping (`patience`):** Setting `patience=0` disables it entirely, keeping the function a drop-in replacement for a fixed-epoch loop. The counter is reset to zero on every AUC improvement so a late recovery still benefits from the full patience budget.
- **In-memory checkpoint rather than disk write:** This keeps the function portable and testable without a file system, and lets the caller batch-write all fold checkpoints or pick only the best across folds.

**Parameters:**
- `model` (`torch.nn.Module`): Network to train; modified in-place throughout.
- `train_loader` (`torch.utils.data.DataLoader`): DataLoader for the training fold.
- `val_loader` (`torch.utils.data.DataLoader`): DataLoader for the validation fold.
- `criterion_cls` (`Type[torch.nn.Module]`): Loss function **class** (not an instance), e.g. `torch.nn.CrossEntropyLoss`. Raises `TypeError` if an instance is passed.
- `optimizer` (`torch.optim.Optimizer`): Configured optimiser.
- `device` (`str`): Target device string.
- `num_epochs` (`int`): Maximum number of training epochs.
- `class_weights` (`Optional[torch.Tensor]`, default `None`): Float tensor of shape `[num_classes]` passed as `weight` to the loss criterion. If `None`, the criterion is instantiated with no weighting.
- `patience` (`int`, default `0`): Early stopping patience in epochs. `0` disables early stopping.
- `grad_accum_steps` (`int`, default `1`): Forwarded to `train_one_epoch`.
- `use_amp` (`bool`, default `True`): Forwarded to both `train_one_epoch` and `validate_one_epoch`.

**Returns:** `Tuple[List[Dict[str, Any]], float, Optional[Dict[str, Any]]]`
- `metrics_history`: List of per-epoch dicts. Each dict contains keys: `"epoch"` (`int`), `"train_loss"` (`float`), `"train_acc"` (`float`), `"val_loss"` (`float`), `"val_acc"` (`float`), `"val_f2"` (`float`), `"val_auc"` (`float`), `"duration_sec"` (`float`).
- `total_training_time_minutes` (`float`): Wall-clock duration of the entire training run in minutes.
- `best_model_checkpoint` (`Optional[Dict[str, Any]]`): Dict with keys `"state_dict"` (deep-copied model weights), `"best_epoch"` (`int`), `"best_val_auc"` (`float`), `"best_val_f2"` (`float`). `None` if no epoch produced a positive AUC (printed warning but no exception raised).

**Side effects:**
- Calls `model.train()` and `model.eval()` on each epoch (via the sub-functions), altering dropout/BatchNorm behaviour.
- Prints a summary line to stdout on every epoch that sets a new best AUC.
- Prints a warning to stdout if no checkpoint was saved.
- Modifies model weights in-place throughout; the returned checkpoint reflects the best-seen state, but the `model` object on the caller side is left in its final (last-epoch) state, not restored to the best checkpoint.

---

### Training loop structure

```
run_training_loop
└── for epoch in range(num_epochs):
    ├── train_one_epoch        → (train_loss, train_acc)
    │   └── for batch in train_loader:
    │       ├── forward (with AMP)
    │       ├── loss.backward() / grad_accum_steps
    │       └── optimizer.step() every grad_accum_steps batches
    ├── validate_one_epoch     → (val_loss, val_acc, val_f2, val_auc)
    │   └── for batch in val_loader:
    │       └── forward (with AMP, torch.no_grad)
    ├── checkpoint if val_auc > best_val_auc   (in-memory deepcopy)
    ├── early stopping check if patience > 0
    └── append epoch dict to metrics_history
```

The epoch dict is always appended, including the epoch that triggers early stopping (the `break` happens after the append in the `else` branch, but the dict is appended inside the `if epochs_without_improvement >= patience` block before the `break`, so it is captured).

---

### Complexity flags

#### 1. Duplicate `metrics_history.append` block

The early-stopping branch at lines 276–286 contains a full copy of the `metrics_history.append(...)` dict (lines 288–297) before calling `break`. This is a direct code duplication: the two dicts are identical. If a new metric key is ever added to the per-epoch record, both sites must be updated in sync.

**Suggested fix:** Move the `append` call unconditionally before the early-stopping check:

```python
metrics_history.append({...})
if patience > 0 and epochs_without_improvement >= patience:
    break
```

This collapses two identical blocks into one and makes the control flow linear.

#### 2. Nested `if patience > 0` / `if epochs_without_improvement >= patience` check

Inside the `else` branch (no improvement), two nested `if` statements guard the early stopping logic. The inner `if` is only reachable when the outer is true, so the two conditions could be written as a single `if patience > 0 and epochs_without_improvement >= patience` — reducing nesting depth by one level and making the intent clearer.

#### 3. `best_model_checkpoint is None` — silent failure mode

If no epoch produces `val_auc > 0.0`, the function returns `None` for the checkpoint. The caller must handle this, but there is no exception — only a `print` warning. A misconfigured run (e.g., a DataLoader that always returns a single class) will silently produce a `None` checkpoint, and downstream code must guard against it. Raising a `RuntimeWarning` or returning an explicit sentinel would make the failure more visible.

#### 4. Model left in last-epoch state after training

The `model` object is modified in-place throughout training, but it is never restored to the best-checkpoint weights before returning. The caller receives the best weights only inside `best_model_checkpoint["state_dict"]` and must explicitly call `model.load_state_dict(...)` if it wants to use the model immediately. This is an implicit contract that is not stated in the docstring and easy to overlook.

---

### Complexity flags summary

| Location | Issue | Suggested simplification |
|---|---|---|
| `run_training_loop` (lines 276–297) | `metrics_history.append` block duplicated inside the early-stopping branch and after it | Move the single `append` call before the early-stopping check; remove the duplicate |
| `run_training_loop` (lines 273–275) | Two nested `if` guards for early stopping (`if patience > 0:` then `if epochs_without_improvement >= patience:`) | Flatten into `if patience > 0 and epochs_without_improvement >= patience:` |
| `run_training_loop` (line 301) | `best_model_checkpoint is None` handled with a `print` warning only | Raise a `RuntimeWarning` or a custom exception to make silent failure more visible |
| `run_training_loop` (post-return) | `model` is left in its last-epoch state; caller must manually load best weights | Document the contract explicitly, or optionally restore `model` to best weights before returning |

---

