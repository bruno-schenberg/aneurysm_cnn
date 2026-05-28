## train_models.py and experiment configs

`training_engine/train_models.py` is the top-level entry point for the training pipeline. It owns argument parsing, experiment config loading and merging, validation, and dispatch to the orchestrator.

---

### CLI interface

The script accepts a single optional argument:

| Argument | Type | Default | What it controls |
|---|---|---|---|
| `--experiments` | `str` (file path) | `experiments.json` | Path to the JSON file that defines the list of experiments to run. Relative paths are resolved against the `training_engine/` directory (i.e. the directory containing the script itself). Absolute paths are used as-is. |

**Usage examples:**

```bash
# Use the default experiments.json (from the training_engine/ directory):
python training_engine/train_models.py

# Point at a different file by relative path:
python training_engine/train_models.py --experiments experiments_192_r3d50.json

# Point at a different file by absolute path:
python training_engine/train_models.py --experiments /mnt/data/my_sweep.json
```

---

### Execution flow

1. **Argument parsing.** `argparse` reads `--experiments` and resolves the path.
2. **JSON loading.** The file is opened and parsed with `json.load`. Two error cases exit immediately: file not found (`SystemExit(1)`) and malformed JSON (`SystemExit(1)`).
3. **Config preparation.** `prepare_experiment_configs` is called on the raw list of dicts. For each entry it: shallow-merges the entry over `DEFAULT_CONFIG`, validates required fields, resolves `data_path` from `data_path_key`, validates tabular config, validates `INPUT_RESOLUTION`, cross-checks that the dataset key and resolution match, and validates `GRAD_ACCUM_STEPS`. Any validation failure raises `ValueError` and halts the process before any training starts.
4. **Sequential execution.** `run_all_experiments` iterates the prepared configs and calls `orchestrator.run_experiment(exp_config)` for each. Errors are caught per-experiment (`FileNotFoundError` and bare `Exception`), logged, and execution continues to the next experiment. `torch.cuda.empty_cache()` is called in the `finally` block after every experiment regardless of outcome.
5. **Final output.** A banner (`EXPERIMENTS FINISHED`) is printed after all experiments have been attempted.

---

### DEFAULT_CONFIG: all keys, types, and defaults

`DEFAULT_CONFIG` is a module-level dict that acts as the schema and source of defaults for every experiment. Keys with `None` as their default are **required** — they must be supplied in the JSON file. All other keys are optional overrides.

#### Required fields (must be present in JSON; no usable default)

| Key | Expected type | Description |
|---|---|---|
| `name` | `str` | Unique experiment identifier; used in output directory names |
| `model` | `str` | Architecture key; must be one of `SUPPORTED_MODELS` in `models.py` |
| `balancing` | `str` | Class imbalance strategy: `"weighted_cost_function"`, `"oversampling"`, or `"none"` |
| `data_path_key` | `str` | Dataset variant key; must be one of the keys in `DATASET_PATHS` (see below) |

#### Derived field (set automatically; must not appear in JSON)

| Key | Description |
|---|---|
| `data_path` | Filesystem path resolved from `data_path_key` via `DATASET_PATHS` |

#### Optional fields (have usable defaults)

| Key | Type | Default | Description |
|---|---|---|---|
| `USE_TABULAR` | `bool` | `False` | Enable late-fusion of age/gender tabular features with CNN features |
| `TABULAR_CSV` | `str` or `None` | `None` | Path to CSV with columns `exam`, `Age`, `gender`; required when `USE_TABULAR` is `True` |
| `DEVICE` | `str` | `"cuda"` if available, else `"cpu"` | Compute device; auto-detected at import time |
| `RANDOM_SEED` | `int` | `42` | Seed for dataset splits, weight init, and shuffle |
| `LEARNING_RATE` | `float` | `0.0001` | Adam/AdamW learning rate |
| `BATCH_SIZE` | `int` | `4` | Training mini-batch size |
| `EPOCHS` | `int` | `2` | Training epochs per fold |
| `N_SPLITS` | `int` | `5` | Number of folds for stratified k-fold CV (ignored when `USE_KFOLD=False`) |
| `USE_KFOLD` | `bool` | `True` | `True` = stratified k-fold CV; `False` = single 70/30-style split |
| `VAL_SPLIT_RATIO` | `float` | `0.30` | Fraction of dev set used for validation when `USE_KFOLD=False` |
| `EARLY_STOPPING_PATIENCE` | `int` | `0` | Epochs without val F2 improvement before stopping; `0` = disabled |
| `WEIGHT_DECAY` | `float` | `1e-4` | AdamW weight-decay (L2 regularisation) |
| `HOLD_OUT_TEST_SET` | `bool` | `True` | Reserve a final held-out test set |
| `TEST_SPLIT_RATIO` | `float` | `0.15` | Fraction of total dataset held out for final evaluation |
| `VAL_BATCH_SIZE` | `int` | `4` | Batch size used during validation and evaluation |
| `CLASSES` | `list[str]` | `["0", "1"]` | Class label strings; `"0"` = healthy, `"1"` = aneurysm |
| `OUTPUT_DIR` | `str` | `"./experiments"` | Root directory for per-experiment output subdirectories |
| `INPUT_RESOLUTION` | `str` | `"128x128x128"` | Target spatial size after resize transform; valid values: `"128x128x128"`, `"192x192x128"`, `"192x192x192"` (via `VALID_RESOLUTIONS` in `data_preprocess`), `"256x256x176"`, `"256x256x256"` |
| `GRAD_ACCUM_STEPS` | `int` | `1` | Gradient accumulation steps; must be `>= 1` and a plain integer |
| `USE_AMP` | `bool` | `True` | Enable automatic mixed precision (bfloat16); auto-disabled on CPU |
| `CACHE_RATE` | `float` | `1.0` | Fraction of dataset to cache in RAM via MONAI `CacheDataset` |

---

### Config loading and merging

Merging is a single shallow dict unpacking on line 161:

```python
config = {**DEFAULT_CONFIG, **exp_json}
```

This means every key from the JSON entry overwrites the corresponding key in `DEFAULT_CONFIG`. There is **no deep merge** — nested objects, if any were used, would replace rather than extend the default. There is also **no base/override inheritance between experiments**: each JSON entry is merged independently against the same fixed `DEFAULT_CONFIG`. Experiments cannot inherit from each other or from a shared "base" block within the JSON file.

---

### DATASET_PATHS: the hardcoded path map

`DATASET_PATHS` maps every valid `data_path_key` to a filesystem path. The root prefix is read from the `DATA_ROOT` environment variable; if that variable is unset the hardcoded fallback `/mnt/data/cases-3` is used.

| Key | Path (relative to `DATA_ROOT`) | Resolution | Description |
|---|---|---|---|
| `A` | `dataset_A_resampled_cropped` | 128³ | 1 mm isotropic resample → crop |
| `B` | `dataset_B_resampled_shrunk` | 128³ | Single-step resample → pad |
| `C` | `dataset_C_cropped` | 128³ | Native resolution → crop |
| `D` | `dataset_D_shrunk` | 128³ | Native resolution → shrink |
| `E` | `dataset_E_fixed_cropped` | 128³ | Fixed spacing → crop |
| `F` | `dataset_F_fixed_shrunk` | 128³ | Fixed spacing → shrink |
| `A192`–`F192` | `dataset_{X}192` | `192x192x128` | 192 × 192 × 128 variants |
| `A256`–`D256` | `dataset_{X}256_*` | `256x256x176` | 256 × 256 × 176 variants |
| `SAMPLE` | `~/sample_dataset` | 128³ | Synthetic dataset for container testing |
| `SAMPLE_D` | `/mnt/data/nifti-sample-dataset-D` | 128³ | 12-case variant-D sample |
| `SAMPLE_A128` | `/mnt/data/nifti-sample-datasets/dataset_A_resampled_cropped` | 128³ | 4-case smoke-test sample |

`DATASET_RESOLUTIONS` enforces that `INPUT_RESOLUTION` in an experiment matches the resolution the chosen dataset key was generated at. A mismatch raises `ValueError` before training begins.

---

### How the orchestrator is invoked

`run_experiment` from `src.orchestrator` is called once per experiment with the single fully-merged config dict:

```python
run_experiment(exp_config)
```

No other arguments are passed. The orchestrator receives the complete config and is responsible for all downstream decisions (data loading, model construction, training loop, evaluation, artifact saving). `train_models.py` does not import or call any other `src.*` module at runtime (only `data_preprocess.VALID_RESOLUTIONS` is imported for validation, and `models.py` is not touched here).

---

### Hardcoded values that should be configurable

| Value | Location | Current form | Issue |
|---|---|---|---|
| `DATA_ROOT` fallback | `train_models.py:79` | `/mnt/data/cases-3` | Hardcoded cluster path; overridable via env var `DATA_ROOT`, but the fallback will silently be wrong in any other environment |
| `SAMPLE_D` path | `DATASET_PATHS` | `/mnt/data/nifti-sample-dataset-D` | Absolute path not derived from `DATA_ROOT`; will break outside the original cluster |
| `SAMPLE_A128` path | `DATASET_PATHS` | `/mnt/data/nifti-sample-datasets/dataset_A_resampled_cropped` | Same issue |
| `MONAI_HOME` | `train_models.py:20` | `~/.cache/monai` via `os.environ.setdefault` | Set unconditionally at import time as a side effect; should be documented or moved to a config |
| `OUTPUT_DIR` default | `DEFAULT_CONFIG` | `"./experiments"` | Relative path resolved from the working directory at runtime, not from the script or repo root; results vary depending on where the script is invoked from |
| Default `EPOCHS` | `DEFAULT_CONFIG` | `2` | Intentionally low for safety, but could be mistaken for a real training value; experiments.json files always override to 100 |

---

### Multi-file experiment JSON structure

There is no central registry of experiment files. The three files in use are:

| File | Experiments | Notes |
|---|---|---|
| `experiments.json` | 8 (R3D18 and R3D50 on datasets B and D, two LRs each) | 128³ baseline sweep; `USE_KFOLD: false`, `VAL_SPLIT_RATIO: 0.20`, `BATCH_SIZE: 8` |
| `experiments_192_r3d50.json` | 6 (R3D50 on datasets A192–F192) | 192 × 192 × 128 sweep; `BATCH_SIZE: 4`, `GRAD_ACCUM_STEPS: 2`, `USE_AMP: false` |
| `experiments_256_baseline.json` | 4 (R3D18 on datasets A256–D256) | 256 × 256 × 176 sweep; `BATCH_SIZE: 2`, `GRAD_ACCUM_STEPS: 4` |

**Patterns and duplication.** Each file is self-contained and repeats the same structural boilerplate across every entry. Within a single file, all experiments share identical values for `balancing`, `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `LEARNING_RATE`, `USE_KFOLD`, `VAL_SPLIT_RATIO`, and `EARLY_STOPPING_PATIENCE` — only `name`, `model`, `data_path_key`, `INPUT_RESOLUTION`, and (in the 192 file) `USE_AMP` vary. This repetition makes updating a shared hyperparameter (e.g. `EPOCHS`) require editing every entry individually.

**No cross-file de-duplication mechanism.** The `--experiments` flag accepts exactly one file per invocation. Running multiple files requires multiple invocations. There is no `"include"` directive or shared-defaults block supported by the loader.

**No experiment-to-experiment inheritance.** Merging always takes `DEFAULT_CONFIG` as the base — a JSON entry cannot say "use everything from experiment X but change Y". Each entry must be fully self-describing.

---

### Complex control flow and simplification opportunities

| Issue | Location | Description |
|---|---|---|
| Silent continue on experiment error | `run_all_experiments` lines 241–244 | Both `FileNotFoundError` and `Exception` are caught, a message is printed, and execution continues. A failed experiment produces no non-zero exit code and no machine-readable error record; downstream consumers (Slurm, CI) cannot detect partial failures. |
| All-or-nothing validation | `prepare_experiment_configs` | Validation runs for all experiments before any training starts, which is good. However, a `ValueError` in validation propagates uncaught through `__main__`, producing a Python traceback rather than a clean error message. Wrapping the call in a `try/except ValueError` with `SystemExit(1)` would match the JSON-load error handling style. |
| Resolution enforcement via two parallel dicts | `DATASET_PATHS` + `DATASET_RESOLUTIONS` | Every dataset key must be registered in both dicts. Adding a new key requires two edits in two places, and a key present in `DATASET_PATHS` but absent from `DATASET_RESOLUTIONS` silently skips the resolution check (`DATASET_RESOLUTIONS.get` returns `None`). A single dict-of-dicts or named tuples would eliminate this split. |
| `DEVICE` evaluated at import time | `DEFAULT_CONFIG` line 41 | `torch.cuda.is_available()` is called when the module is imported, not when an experiment runs. This is harmless in practice but means the default cannot be changed by code that modifies the environment after import. |
| Repeated boilerplate in JSON files | All three experiment files | A "defaults" section at the top of each JSON file (or a shared `base` object that experiments reference) would eliminate per-entry repetition and make sweeps easier to maintain. This would require extending the loader in `prepare_experiment_configs`.

---

