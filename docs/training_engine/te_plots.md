## plots.py

`training_engine/src/plots.py` contains plotting, metric-calculation, and artifact-saving utilities. It is called by the orchestrator at the end of each fold to produce a complete record of training behaviour and evaluation results.

**Plotting library:** `matplotlib.pyplot` (aliased as `plt`). Supporting libraries: `numpy`, `pandas`, `scikit-learn` (metrics and `roc_curve`/`auc`), `torch`, `csv`.

---

### Functions

---

#### `plot_confusion_matrix`

**What it does:** Renders a confusion matrix as a colour-mapped grid and saves it to a PNG file.

| Parameter | Type | Description |
|---|---|---|
| `cm` | `np.ndarray` | Confusion matrix of shape `(n_classes, n_classes)`. |
| `classes` | `List[str]` | Class label strings used for axis tick labels. |
| `filename` | `str` | Destination file path for the saved PNG. |
| `normalize` | `bool` | If `True`, row-normalises so each cell shows the fraction of true-class samples predicted in that column. Default `False`. |
| `title` | `str` | Title string rendered above the matrix. Default `"Confusion Matrix"`. |

**Returns:** `None`

**Writes to disk:** A single PNG at the path provided in `filename`.

**Data source:** Receives the pre-computed matrix array directly — does **not** read from disk. Preferred for testability.

**Flags:**
- Hardcoded figure size `(6, 6)` and colourmap `plt.cm.Blues`.
- Cell text colour threshold `cm.max() / 2.0` is an inline constant with no named symbol.

---

#### `plot_roc_curve`

**What it does:** Re-runs inference over a DataLoader to collect class probabilities, then plots and saves a ROC curve PNG.

| Parameter | Type | Description |
|---|---|---|
| `model` | `torch.nn.Module` | Trained model (set to `eval()` internally). |
| `dataloader` | `torch.utils.data.DataLoader` | DataLoader over the evaluation split. |
| `device` | `str` | PyTorch device string (`"cuda"` or `"cpu"`). |
| `classes` | `List[str]` | Class label strings (accepted but not used in the plot body; present for API symmetry). |
| `filename` | `str` | Destination file path for the saved PNG. |

**Returns:** `None` (returns early without writing if only one class is found in the labels).

**Writes to disk:** A single PNG at the path provided in `filename`.

**Data source:** **Reads from a live DataLoader** — triggers a full forward pass over the dataset. This is the only function in the module that re-runs inference; all others operate on pre-collected prediction records. This makes it harder to test in isolation and couples evaluation cost to artifact generation.

**Flags:**
- Batch dict keys `"image"` and `"label"` are hardcoded string literals; any DataLoader that uses different keys will silently fail.
- Branch at `outputs.shape[1] == 2` selects between softmax-class-1 probability and raw squeeze output; the scalar-output path (`squeeze(1)`) has no activation applied, so it assumes logits already represent a probability.
- Hardcoded line colours (`"darkorange"`, `"navy"`) and axis limits (`[0.0, 1.05]` on y).

---

#### `calculate_classification_metrics`

**What it does:** Computes Precision, Recall, and F2-Score from a list of per-sample prediction records.

| Parameter | Type | Description |
|---|---|---|
| `detailed_results` | `List[Dict[str, Any]]` | Each dict must contain integer-valued `"true_label"` and `"pred_label"` keys. |

**Returns:** `Dict[str, float]` with keys `"Precision"`, `"Recall"`, `"F2-Score"`. Returns `NaN` for all keys on empty input or on exception; returns `0.0` for all keys when only one class is present.

**Writes to disk:** Nothing.

**Data source:** Receives data directly as arguments. Preferred for testability.

**Flags:**
- Bare `except Exception` swallows all errors and returns `NaN` values silently (only a `print` statement is emitted); exceptions from malformed dicts will be invisible to callers.
- `pos_label=1` is hardcoded, so the metrics are always anchored to class index 1 as the positive class; binary datasets where the aneurysm label is `0` would produce incorrect results.

---

#### `evaluate_models`

**What it does:** Aggregates per-fold evaluation metrics across all folds and saves a summary CSV that includes per-fold rows plus an average and standard-deviation footer.

| Parameter | Type | Description |
|---|---|---|
| `all_fold_predictions` | `Dict[str, Dict[str, Any]]` | Maps fold/model name strings to fold result dicts; each inner dict must contain `"metrics"` (a dict), `"eval_acc"` (float), and `"eval_loss"` (float). |
| `output_csv_path` | `str` | Destination file path for the summary CSV. |

**Returns:** `None`

**Writes to disk:** A CSV at `output_csv_path` with columns `Model Name`, `Precision`, `Recall`, `F2-Score`, `Eval Accuracy`, `Eval Loss`, followed by an `"Average"` row and a `"Std. Dev."` row.

**Data source:** Receives data directly as arguments. Preferred for testability.

**Flags:**
- Column names (`"Precision"`, `"Recall"`, `"F2-Score"`, `"Eval Accuracy"`, `"Eval Loss"`) are duplicated as string literals in the dict-construction block and in the explicit average/std blocks; if `calculate_classification_metrics` ever renames a key the two sites will silently diverge.
- Silently prints "No valid fold data was processed." and writes nothing if all folds lack a `"metrics"` key; the caller receives no programmatic signal of failure.

---

#### `save_predictions_from_detailed_results`

**What it does:** Writes per-sample prediction records to a CSV file for post-hoc analysis.

| Parameter | Type | Description |
|---|---|---|
| `detailed_results` | `List[Dict[str, Any]]` | Each dict must contain `"filename"`, `"true_label"`, and `"pred_label"` keys. |
| `filename` | `str` | Destination file path for the CSV. |

**Returns:** `None`

**Writes to disk:** A CSV at `filename` with columns `file_name`, `true_label`, `prediction`.

**Data source:** Receives data directly as arguments. Preferred for testability.

**Flags:**
- Source key `"filename"` (no underscore) maps to output column `"file_name"` (with underscore); the asymmetry could cause confusion when joining this output against other files that use `"filename"`.
- No error handling for missing keys in individual records; a malformed dict will raise `KeyError` at the list-comprehension stage.

---

#### `save_metrics_to_csv`

**What it does:** Concatenates the per-epoch training history with a final-evaluation metrics row and saves the result to a single CSV file.

| Parameter | Type | Description |
|---|---|---|
| `metrics_history` | `List[Dict[str, Any]]` | Per-epoch metric dicts (expected to contain at least an `"epoch"` key plus loss/accuracy columns). |
| `filename` | `str` | Destination file path for the CSV. |
| `detailed_results` | `List[Dict[str, Any]]` | Per-sample prediction dicts forwarded to `calculate_classification_metrics`. |

**Returns:** `None`

**Writes to disk:** A CSV at `filename` combining epoch rows and a trailing `"final_eval"` row containing Precision, Recall, and F2-Score. Floats are formatted to 4 decimal places.

**Data source:** Receives data directly as arguments. Preferred for testability. Internally calls `calculate_classification_metrics`.

**Flags:**
- If `metrics_history` is empty the function returns early with only a `print` warning and no file is written.
- The `"final_eval"` row will have `NaN` or `0.0` in metric columns if `detailed_results` is empty or single-class; this is silently inherited from `calculate_classification_metrics`.

---

#### `save_fold_artifacts`

**What it does:** Orchestrates the generation and saving of all evaluation artifacts for one completed fold: raw and normalised confusion-matrix PNGs, a ROC-curve PNG, a predictions CSV, a metrics CSV, and two model checkpoint files (last epoch and best epoch).

| Parameter | Type | Description |
|---|---|---|
| `model` | `torch.nn.Module` | Trained model at the last epoch. |
| `eval_loader` | `torch.utils.data.DataLoader` | DataLoader over the test/validation split. |
| `best_model_state` | `Dict[str, Any]` | State dict returned by `run_training_loop`; if falsy, no best-model checkpoint is written. |
| `detailed_results` | `List[Dict[str, Any]]` | Per-sample prediction records. |
| `metrics_history` | `List[Dict[str, Any]]` | Per-epoch metric dicts. |
| `experiment_name` | `str` | Unique experiment identifier used as a filename prefix. |
| `fold` | `int` | 1-based fold index used in all filenames. |
| `fold_output_dir` | `str` | Directory under which all artifact files are written. |
| `config` | `Dict[str, Any]` | Experiment config dict; must contain `"CLASSES"` (`List[str]`) and `"DEVICE"` (`str`). |

**Returns:** `None`

**Writes to disk (all under `fold_output_dir`):**

| File | Format | Description |
|---|---|---|
| `{experiment_name}_fold{fold}_cm_raw.png` | PNG | Unnormalised confusion matrix. |
| `{experiment_name}_fold{fold}_cm_normalized.png` | PNG | Row-normalised confusion matrix. |
| `{experiment_name}_fold{fold}_roc_curve.png` | PNG | ROC curve. |
| `{experiment_name}_fold{fold}_predictions.csv` | CSV | Per-sample predictions. |
| `{experiment_name}_fold{fold}_metrics.csv` | CSV | Epoch history + final eval row. |
| `{experiment_name}_fold{fold}_model_last.pth` | PyTorch state dict | Weights at the final training epoch. |
| `{experiment_name}_fold{fold}_model_best.pth` | PyTorch state dict | Weights at the best-checkpoint epoch (only if `best_model_state` is truthy). |

**Data source:** Mixes both patterns — pre-computed `detailed_results` are used for confusion matrices and CSVs without re-running the model, but `plot_roc_curve` triggers a full second forward pass over `eval_loader`. This asymmetry means evaluation cost is not uniform across artifact types.

**Flags:**
- All seven output paths are constructed with `os.path.join` using a consistent naming convention; no hardcoded absolute paths.
- `config["CLASSES"]` and `config["DEVICE"]` are accessed without `.get()` defaults; missing keys will raise `KeyError`.
- No return value and no summary of what was written; the caller cannot programmatically verify which files were produced.
- `best_model_path` is computed unconditionally even when `best_model_state` is falsy, wasting the `os.path.join` call (minor).

---

### Module-level flags summary

| Flag | Detail |
|---|---|
| Hardcoded dict keys for DataLoader batches | `"image"` and `"label"` in `plot_roc_curve` — coupling to a specific MONAI-style dataset convention |
| Hardcoded `pos_label=1` | In `calculate_classification_metrics` — assumes aneurysm is always class index 1 |
| Re-inference in artifact generation | `plot_roc_curve` re-runs the full forward pass; all other functions use pre-collected records |
| Bare `except Exception` | In `calculate_classification_metrics` — swallows unexpected errors silently |
| No programmatic failure signals | `evaluate_models` and `save_fold_artifacts` only `print` on failure; callers cannot distinguish success from silent no-ops |

---

