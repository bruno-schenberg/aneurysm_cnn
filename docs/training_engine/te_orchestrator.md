## orchestrator.py

`training_engine/src/orchestrator.py` is the top-level coordinator for a full cross-validation experiment. It owns the fold loop, wires together every other module, and is the only place where the sequence `data split → model init → train → evaluate → save → aggregate` is expressed end-to-end.

---

### Module-level state and implicit ordering

There is **no module-level mutable state**. All state flows through the `config` dict and local variables.

There is one **implicit ordering requirement**: `prepare_experiment_setup` must be called before the fold loop because it calls `set_seed`, creates the output directory, and returns the `fold_splits` generator. If the fold loop were entered without calling it first, the split would be non-reproducible and the output directory would not exist. This contract is enforced only by the call order inside `run_experiment`; nothing prevents a caller from skipping `prepare_experiment_setup` and calling `run_one_fold` directly.

There is one **implicit environment dependency**: the `EXPERIMENT_BACKUP_DIR` environment variable is read at the very end of `run_experiment`. Its presence or absence changes observable behavior (a `shutil.copytree` call), but it is never documented in the function signature or docstring of `run_experiment`.

---

### How experiments are loaded from JSON config

`orchestrator.py` does **not** read JSON directly. It receives an already-parsed `config: Dict[str, Any]` dict. The JSON loading and merging of per-experiment overrides with global defaults is done upstream by the caller (`train_models.py`, not present in `src/`). The orchestrator treats `config` as a fully-formed, validated dict and accesses keys directly (e.g., `config["LEARNING_RATE"]`), using `.get(key, default)` only for optional keys. There is no schema validation inside the orchestrator; a missing required key raises a `KeyError` at runtime.

---

### How the orchestrator dispatches training runs

`run_experiment` is the single dispatch point. It is called once per experiment by the external caller. Internally it iterates over `fold_splits` (a generator of `(fold_idx, train_files, val_files, test_files)` tuples) and delegates each fold to `run_one_fold`. There is no dynamic dispatch table; the orchestrator always runs `run_one_fold` for every fold in sequence. The only conditional branching at the experiment level is the two-line balancing resolution (`oversample` / `use_class_weights`) and the optional backup copy at the end.

---

### Functions

---

#### `run_one_fold`

**What it does:** Executes one complete cross-validation fold — optimiser init, training loop, checkpoint restoration, evaluation, artifact saving — and returns a result dict for aggregation.

**Design rationale:** Encapsulating one fold as a single function keeps `run_experiment`'s fold loop short and makes it straightforward to test a fold in isolation. The function intentionally does not mutate any shared state; every artefact is written under the fold-specific `fold_output_dir`.

**Input parameters:**

| Parameter | Type | Description |
|---|---|---|
| `fold` | `int` | 1-based fold index used in print messages and file names |
| `train_loader` | `torch.utils.data.DataLoader` | Training data for this fold |
| `val_loader` | `torch.utils.data.DataLoader` | Validation data for this fold |
| `test_loader` | `torch.utils.data.DataLoader` | Hold-out test set, identical across folds |
| `model` | `torch.nn.Module` | Freshly initialised model already moved to the target device |
| `config` | `Dict[str, Any]` | Fully-formed experiment config dict |
| `weights_tensor` | `Optional[torch.Tensor]` | Per-class loss weights, or `None` |
| `experiment_output_dir` | `str` | Root output directory; a `fold_{n}` subdirectory is created inside it |

**Return value:** `Dict[str, Any]` — a FoldResult dict with keys `'predictions'` (list of per-sample detail dicts from `validate_one_epoch`), `'metrics'` (dict from `calculate_classification_metrics`), `'eval_acc'` (float), and `'eval_loss'` (float).

**Coordination with other modules:**

- `training.run_training_loop` — trains the model and returns `(metrics_history, total_time, best_model_checkpoint)`.
- `training.validate_one_epoch` — evaluates the restored checkpoint on the eval set with an unweighted `CrossEntropyLoss`; `return_details=True` is always passed to obtain per-sample predictions.
- `plots.save_fold_artifacts` — writes plots, CSVs, and model weights under `fold_output_dir`.
- `plots.calculate_classification_metrics` — converts the raw `detailed_results` list into a human-readable metrics dict used for the console summary line.

**Control flow notes:**

- There is a **guard** on `best_model_checkpoint` being `None` (line 114–117): the code falls back to final-epoch weights and prints a warning. The same guard is repeated two more times (lines 137 and 155) to conditionally write the metadata JSON and extract the state dict. This triple-check for the same `None` condition is minor duplication; extracting `(best_epoch, best_val_auc, best_val_f2, best_state_dict)` into local variables once at line 114 would eliminate all three branches.
- The `HOLD_OUT_TEST_SET` branch (lines 120–125) is a simple two-way switch (`test_loader` vs `val_loader`). It is clear but could be simplified to a single ternary assignment.

---

#### `prepare_experiment_setup`

**What it does:** Validates the experiment config, seeds the RNG, creates the output directory, loads the full dataset file list, and returns a fold-split generator — all before any GPU work begins.

**Design rationale:** Separating setup from training means any config or filesystem error surfaces immediately, before GPU memory is allocated. Calling `set_seed` here (rather than inside each fold) ensures the k-fold split itself is reproducible while still allowing each fold to produce a different partition.

**Input parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `Dict[str, Any]` | Fully-formed experiment config dict |

**Return value:** `tuple[str, Any]` — `(experiment_output_dir, fold_splits)` where `fold_splits` is a generator yielding `(fold_idx, train_files, val_files, test_files)` tuples.

**Raises:** `FileNotFoundError` if `config['data_path']` does not exist.

**Coordination with other modules:**

- `data_preprocess.set_seed` — seeds Python, NumPy, and PyTorch RNGs.
- `data_preprocess.get_data_list` — scans `data_path` (and optionally reads a tabular CSV) to produce the full file list.
- `data_preprocess.split_data` — partitions the file list into k-fold or single-split train/val/test generators, respecting `USE_KFOLD`, `N_SPLITS`, `TEST_SPLIT_RATIO`, `VAL_SPLIT_RATIO`, and `RANDOM_SEED`.

**Implicit ordering requirement:** Must be called before the fold loop. `set_seed` has a global side effect on the Python/NumPy/PyTorch RNG state; any random operation between this call and the fold loop would shift the split.

---

#### `summarize_experiment_results`

**What it does:** Aggregates per-fold result dicts and writes the experiment-level summary CSV (one row per fold plus Average and Std. Dev. rows).

**Design rationale:** Thin wrapper around `plots.evaluate_models` that keeps the summary-writing concern out of `run_experiment`'s fold loop. Its only job is constructing the CSV path and delegating.

**Input parameters:**

| Parameter | Type | Description |
|---|---|---|
| `experiment_name` | `str` | Unique experiment identifier, used to name the output CSV |
| `experiment_output_dir` | `str` | Root output directory for the experiment |
| `all_fold_predictions` | `Dict[str, Any]` | Maps fold names (`"{name}_fold{n}"`) to FoldResult dicts |

**Return value:** `None`

**Coordination with other modules:** Delegates entirely to `plots.evaluate_models`.

---

#### `run_experiment`

**What it does:** Runs a complete cross-validation experiment end-to-end: setup, fold loop (data loading → model init → train → evaluate → save), and post-experiment summarisation and optional backup.

**Design rationale:** Single entry point called once per experiment by the external launcher. Keeps the fold loop short by delegating all per-fold logic to `run_one_fold`. The model is reinitialised from scratch each fold so performance differences are attributable to the data split, not accumulated training state. `del model` + `torch.cuda.empty_cache()` between folds prevents temporarily holding two large models in VRAM simultaneously.

**Input parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `Dict[str, Any]` | Fully-formed experiment config dict as returned by the external launcher |

**Return value:** `None`

**Coordination with other modules:**

| Module | Functions called | Purpose |
|---|---|---|
| `data_preprocess` | `set_seed`, `get_data_list`, `split_data` (via `prepare_experiment_setup`); `build_dataloaders`, `get_class_weights`, `parse_input_resolution` | Data splitting, loader construction, class-weight computation, resolution parsing |
| `models` | `get_model` | Model construction and, for multi-modal configs, wrapping |
| `training` | `run_training_loop`, `validate_one_epoch` (via `run_one_fold`) | Training loop execution and final evaluation |
| `plots` | `save_fold_artifacts`, `calculate_classification_metrics`, `evaluate_models` (via `run_one_fold` and `summarize_experiment_results`) | Artifact saving and result aggregation |

**Control flow — fold loop (lines 318–362):**

```
for fold_idx, train_files, val_files, test_files in fold_splits:
    build_dataloaders(...)
    [optionally] get_class_weights(...)
    get_model(...)
    model.to(device)
    run_one_fold(...)
    del model
    [optionally] cuda.empty_cache()
    all_fold_predictions[fold_name] = fold_results
```

This is straightforward and flat — no nesting beyond the single `for` loop. No complexity concern.

**Balancing dispatch (lines 310–311):**

```python
oversample = balancing == "oversampling"
use_class_weights = balancing == "weighted_cost_function"
```

This resolves the `balancing` string field into two boolean flags. It correctly handles three states: `"oversampling"`, `"weighted_cost_function"`, and any other value (both flags `False`, meaning no balancing). If a fourth balancing strategy were added, this pattern would require a third flag line. A cleaner alternative would be a small dispatch dict:

```python
# Proposed alternative
_BALANCING = {
    "oversampling": dict(oversample=True, use_class_weights=False),
    "weighted_cost_function": dict(oversample=False, use_class_weights=True),
}
flags = _BALANCING.get(balancing, dict(oversample=False, use_class_weights=False))
oversample, use_class_weights = flags["oversample"], flags["use_class_weights"]
```

---

### Global state and implicit dependencies

| Item | Nature | Risk |
|---|---|---|
| `EXPERIMENT_BACKUP_DIR` env variable | Read at the end of `run_experiment` via `os.environ.get` | Silent no-op if unset; no warning; undocumented in the function signature |
| `set_seed` RNG side effect | Global mutation of Python/NumPy/PyTorch RNG state inside `prepare_experiment_setup` | Any random operation between `prepare_experiment_setup` and the first fold iteration would shift the reproducible split |
| `del model` + `cuda.empty_cache()` ordering | Must happen after `run_one_fold` completes and before the next `get_model` call | Enforced by sequential loop order; no guard if `run_one_fold` raises mid-fold (model would stay allocated until the exception propagates) |

---

### Complexity flags

| Location | Issue | Suggested simplification |
|---|---|---|
| `run_one_fold` (lines 114, 137, 155) | `best_model_checkpoint is None` checked three separate times | Extract `best_epoch`, `best_val_auc`, `best_val_f2`, `best_state_dict` into local variables once at line 114; replace subsequent branches with direct variable references |
| `run_one_fold` (lines 120–125) | Two-branch `if/else` for eval loader selection | Collapse to a single ternary: `eval_loader = test_loader if config.get("HOLD_OUT_TEST_SET", False) else val_loader` |
| `run_experiment` (lines 310–311) | Two parallel boolean flags derived from one string field | Replace with a dispatch dict (see proposed alternative above) to make adding new balancing strategies trivially safe |
| `run_one_fold` (line 117) | `None` checkpoint silently falls back to final-epoch weights with only a `print` warning | Promote to `warnings.warn(..., RuntimeWarning)` so callers that set `PYTHONWARNINGS=error` can catch it |

---

