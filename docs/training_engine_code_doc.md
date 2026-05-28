## Pipeline Overview

This section traces a complete training run from the CLI invocation all the way to the files written to disk, naming every function and data structure that crosses a module boundary.

### Entry point — `train_models.py`

```
python training_engine/train_models.py --experiments experiments_192_r3d50.json
```

1. `argparse` resolves the path and calls `json.load` → `experiments_to_run: List[Dict]`.
2. `prepare_experiment_configs(experiments_to_run)` iterates the raw list and for each entry:
   - shallow-merges `{**DEFAULT_CONFIG, **exp_json}` into `config: Dict[str, Any]`
   - validates required fields (`name`, `model`, `balancing`, `data_path_key`)
   - resolves `config["data_path"]` from `DATASET_PATHS[data_path_key]`
   - cross-checks `INPUT_RESOLUTION` against `DATASET_RESOLUTIONS[data_path_key]`
   - appends the validated `config` to `prepared_configs: List[Dict[str, Any]]`
3. `run_all_experiments(prepared_configs)` loops over the list and calls `run_experiment(exp_config)` once per experiment, catching per-experiment exceptions so the run continues.

The fully-merged `config` dict is the single data structure passed to every downstream layer. No other arguments are threaded through.

---

### Orchestrator — `src/orchestrator.py`

`run_experiment(config)` sequences the entire cross-validation experiment:

**Phase 1 — `prepare_experiment_setup(config)`**

- Calls `set_seed(config["RANDOM_SEED"])` — seeds Python `random`, NumPy, PyTorch CPU/CUDA, and MONAI determinism. This is a global side effect; it must happen before any data split.
- Creates `experiment_output_dir = config["OUTPUT_DIR"] / config["name"]` with `os.makedirs`.
- Calls `get_data_list(config["data_path"], tabular_csv=config.get("TABULAR_CSV"))` → `all_data: List[Dict]`. Each dict has keys `"image"` (absolute NIfTI path), `"label"` (0 or 1), `"path"` (filename), and optionally `"tabular"` (float32 array `[age/100, gender]`).
- Calls `split_data(all_data, ...)` → `fold_splits`: a generator that yields `(fold_idx: int, train_files: List[Dict], val_files: List[Dict], test_files: List[Dict])` tuples. The held-out test set is identical across all folds; only the train/val partition changes.

**Phase 2 — fold loop**

For each `(fold_idx, train_files, val_files, test_files)` from the generator:

1. `parse_input_resolution(config["INPUT_RESOLUTION"])` → `spatial_size: Tuple[int, int, int]` (e.g. `(192, 192, 128)`).
2. `build_dataloaders(train_files, val_files, test_files, ..., spatial_size=spatial_size, cache_rate=config["CACHE_RATE"])` → `(train_loader, val_loader, test_loader)`. Each loader yields batches of `{"image": Tensor(B,1,D,H,W), "label": Tensor(B,), "path": List[str]}`.
3. If `balancing == "weighted_cost_function"`: `get_class_weights(train_files)` → `weights_tensor: torch.Tensor([w0, w1])`. Computed from the training fold only to prevent label leakage.
4. `get_model(config["model"], num_classes=2, use_tabular=..., spatial_size=spatial_size)` → `model: nn.Module`. Model is moved to `config["DEVICE"]`.
5. `run_one_fold(fold_idx+1, train_loader, val_loader, test_loader, model, config, weights_tensor, experiment_output_dir)` — runs one fold and returns `fold_results: Dict`.
6. `del model; torch.cuda.empty_cache()` — explicitly releases VRAM before the next fold allocates a fresh model.
7. Fold result stored: `all_fold_predictions[f"{name}_fold{fold_idx+1}"] = fold_results`.

**Phase 3 — `summarize_experiment_results(...)`**

Calls `evaluate_models(all_fold_predictions, summary_csv_path)`. Writes one CSV to disk:
- `{experiment_output_dir}/{experiment_name}_evaluation_summary.csv` — one row per fold plus `Average` and `Std. Dev.` rows.

**Optional backup**: if `EXPERIMENT_BACKUP_DIR` env var is set, `shutil.copytree` copies the entire experiment output directory there.

---

### Data preprocessing — `src/data_preprocess.py`

`build_dataloaders` builds the MONAI transform pipelines by calling `get_transforms`:

- **Training**: `get_transforms(spatial_size, augment=True)` → `Compose([LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, RandFlipd×3, RandRotate90d, RandRotated, RandScaleIntensityd, RandZoomd, RandGaussianNoised, EnsureTyped])`.
- **Val/test**: same chain without the `Rand*` transforms.

Datasets are wrapped in `CacheDataset` (when `cache_rate > 0`) or plain `Dataset`. `CacheDataset` applies deterministic transforms (load through scale) once and caches the result; stochastic augmentations are re-applied live each epoch. The output of every dataset item is a dict with keys `"image"` (`Tensor` shape `(1, D, H, W)`), `"label"` (`int` tensor), and `"path"` (`str`).

---

### Model — `src/models.py`

`get_model(model_name, num_classes, use_tabular, spatial_size)` dispatches through an `if/elif` chain to one of the per-architecture constructors (e.g. `get_r3d18_monai_model`, `get_densenet121_monai_model`). All returned models share the interface: input `(B, 1, D, H, W)`, output logits `(B, num_classes)`. When `use_tabular=True`, `_strip_classifier` removes the final linear layer and the model is wrapped in `MultiModalWrapper`, which concatenates CNN features with the `"tabular"` tensor before a new classifier head.

---

### Training loop — `src/training.py`

`run_one_fold` in the orchestrator sets up the optimiser and delegates to:

```
run_training_loop(model, train_loader, val_loader, criterion_cls=nn.CrossEntropyLoss,
                  optimizer, device, num_epochs, class_weights, patience, grad_accum_steps, use_amp)
  → (metrics_history: List[Dict], total_time_minutes: float, best_model_checkpoint: Optional[Dict])
```

The function instantiates the criterion internally (so `class_weights.to(device)` is correct), then for each epoch:

1. `train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum_steps, use_amp)` → `(train_loss, train_acc)`. Forward pass uses `torch.amp.autocast(bfloat16)`; gradients are accumulated over `grad_accum_steps` batches before `optimizer.step()`.
2. `validate_one_epoch(model, val_loader, criterion, device, use_amp=use_amp)` → `(val_loss, val_acc, val_f2, val_auc)`. Runs under `torch.no_grad()`.
3. If `val_auc > best_val_auc`: stores `{"state_dict": copy.deepcopy(model.state_dict()), "best_epoch": epoch+1, "best_val_auc": val_auc, "best_val_f2": val_f2}` in `best_model_checkpoint`. Checkpoint is kept purely in memory — no file write here.
4. Appends `{"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f2", "val_auc", "duration_sec"}` to `metrics_history`.
5. Early-stopping: if `patience > 0` and `epochs_without_improvement >= patience`, breaks.

After the loop, `run_one_fold` restores the best weights with `model.load_state_dict(best_model_checkpoint["state_dict"])`, then calls `validate_one_epoch(..., return_details=True)` on the eval set (test or val, controlled by `HOLD_OUT_TEST_SET`) to produce `detailed_results: List[Dict]` — one dict per sample with keys `"filename"`, `"true_label"`, `"pred_label"`.

---

### Checkpoints and plots — `src/plots.py`

`save_fold_artifacts(model, eval_loader, best_model_state, detailed_results, metrics_history, ...)` writes all fold-level artifacts under `{experiment_output_dir}/fold_{n}/`:

| File written | Source function | Data source |
|---|---|---|
| `{name}_fold{n}_cm_raw.png` | `plot_confusion_matrix` | `detailed_results` (no re-inference) |
| `{name}_fold{n}_cm_normalized.png` | `plot_confusion_matrix` | `detailed_results` (no re-inference) |
| `{name}_fold{n}_roc_curve.png` | `plot_roc_curve` | re-runs forward pass over `eval_loader` |
| `{name}_fold{n}_predictions.csv` | `save_predictions_from_detailed_results` | `detailed_results` |
| `{name}_fold{n}_metrics.csv` | `save_metrics_to_csv` | `metrics_history` + `detailed_results` |
| `{name}_fold{n}_model_last.pth` | `torch.save` | `model.state_dict()` (last epoch) |
| `{name}_fold{n}_model_best.pth` | `torch.save` | `best_model_state` (AUC-optimal epoch) |
| `best_checkpoint_metadata.json` | `json.dump` (in `run_one_fold`) | `best_epoch`, `best_val_auc`, `best_val_f2` |

`best_model_best.pth` is a bare state dict (not the full checkpoint dict) and can be loaded directly with `model.load_state_dict(torch.load(...))`. `best_checkpoint_metadata.json` is the only file that records which epoch was selected and why.

---

### Complete call graph (one fold)

```
train_models.py
└── run_all_experiments
    └── run_experiment(config)                         [orchestrator]
        ├── prepare_experiment_setup(config)
        │   ├── set_seed(seed)                         [data_preprocess]
        │   ├── get_data_list(data_path)               [data_preprocess] → List[Dict]
        │   └── split_data(all_data, ...)              [data_preprocess] → generator
        └── for fold in fold_splits:
            ├── build_dataloaders(...)                 [data_preprocess]
            │   └── get_transforms(spatial_size, augment) → Compose
            ├── get_class_weights(train_files)         [data_preprocess] → Tensor|None
            ├── get_model(model_name, ...)             [models]          → nn.Module
            └── run_one_fold(...)                      [orchestrator]
                ├── run_training_loop(...)             [training]
                │   ├── for epoch:
                │   │   ├── train_one_epoch(...)       [training]
                │   │   └── validate_one_epoch(...)    [training]
                │   └── → (metrics_history, time, best_checkpoint)
                ├── validate_one_epoch(return_details) [training] → detailed_results
                ├── json.dump(best_checkpoint_metadata)           → fold_n/best_checkpoint_metadata.json
                └── save_fold_artifacts(...)           [plots]
                    ├── plot_confusion_matrix(...)                → cm_raw.png, cm_normalized.png
                    ├── plot_roc_curve(...)                       → roc_curve.png  (re-inference)
                    ├── save_predictions_from_detailed_results    → predictions.csv
                    ├── save_metrics_to_csv(...)                  → metrics.csv
                    └── torch.save(state_dict)                    → model_last.pth, model_best.pth
    └── summarize_experiment_results(...)              [orchestrator]
        └── evaluate_models(all_fold_predictions)      [plots]   → evaluation_summary.csv
```

---

## Architectural Suggestions

### 1. Experiment config organisation

**Current state.** There are currently five experiment JSON files in `training_engine/`:

| File | Experiments | Repeated boilerplate per entry |
|---|---|---|
| `experiments.json` | 8 | `balancing`, `EPOCHS`, `BATCH_SIZE`, `LR`, `USE_KFOLD`, `VAL_SPLIT_RATIO`, `EARLY_STOPPING_PATIENCE` |
| `experiments_192_r3d50.json` | 6 | same + `INPUT_RESOLUTION`, `GRAD_ACCUM_STEPS`, `USE_AMP` |
| `experiments_256_baseline.json` | 4 | same as 192 |
| `experiments_128_baseline.json` | varies | same pattern |
| `experiments_smoke_test.json` | 10 | `balancing`, `EPOCHS`, `BATCH_SIZE`, `LR`, `USE_AMP`, `USE_KFOLD`, `HOLD_OUT_TEST_SET`, `VAL_SPLIT_RATIO`, `EARLY_STOPPING_PATIENCE`, `GRAD_ACCUM_STEPS` |

Every entry within a file repeats an identical block of 7–11 fields. Changing `EARLY_STOPPING_PATIENCE` from 10 to 15 across a file requires editing every entry individually.

**Proposed layout.**

```
training_engine/
    configs/
        base_128.json            # shared defaults for 128³ experiments
        base_192.json            # shared defaults for 192 experiments
        base_256.json            # shared defaults for 256 experiments
        base_smoke.json          # shared defaults for smoke-test runs
        r3d18_128_sweep.json     # experiment list for 128³ R3D18 sweep
        r3d50_192_sweep.json     # experiment list for 192 R3D50 sweep
        r3d18_256_sweep.json     # experiment list for 256 R3D18 baseline
        smoke_test.json          # all smoke-test experiments
```

Each experiment file would reference a base config by key and list only the fields that differ:

```json
{
    "base": "base_192",
    "experiments": [
        {"name": "R3D50_A192_lr1e4", "model": "R3D50", "data_path_key": "A192"},
        {"name": "R3D50_B192_lr1e4", "model": "R3D50", "data_path_key": "B192"}
    ]
}
```

A base file contains the shared fields:

```json
{
    "balancing": "weighted_cost_function",
    "EPOCHS": 100,
    "BATCH_SIZE": 4,
    "GRAD_ACCUM_STEPS": 2,
    "LEARNING_RATE": 0.0001,
    "INPUT_RESOLUTION": "192x192x128",
    "USE_AMP": false,
    "USE_KFOLD": false,
    "VAL_SPLIT_RATIO": 0.20,
    "EARLY_STOPPING_PATIENCE": 10
}
```

**Loader change in `prepare_experiment_configs`.**  The change to `train_models.py` is minimal. After loading the JSON, detect whether the top-level object is a list (current format, backward-compatible) or a dict with `"base"` and `"experiments"` keys (new format):

```python
if isinstance(raw, list):
    raw_experiments = raw          # current flat-list format
else:
    base_file = os.path.join(configs_dir, raw["base"] + ".json")
    with open(base_file) as f:
        base = json.load(f)
    raw_experiments = [{**base, **exp} for exp in raw["experiments"]]
```

The merged list is then passed to `prepare_experiment_configs` unchanged. No other module needs to be touched.

**CLI.** The `--experiments` flag already accepts any path, so no CLI change is needed. Running the 192 sweep becomes:

```bash
python training_engine/train_models.py --experiments configs/r3d50_192_sweep.json
```

**Naming convention for experiment files.**

```
{model_family}_{resolution}_{theme}.json   # e.g. r3d50_192_sweep.json
{theme}.json                               # e.g. smoke_test.json
base_{resolution}.json                     # e.g. base_192.json
```

---

### 2. Result fetching from cluster

The following Makefile target (or equivalent shell script) provides a concrete, repeatable workflow. It assumes the cluster is accessible as `cluster` (either a Host entry in `~/.ssh/config` or a direct `user@hostname` string), that jobs write results under `$RESULTS_ROOT` on the cluster, and that the local mirror lives at `./results/`.

```makefile
# ── Cluster sync ──────────────────────────────────────────────────────────────
CLUSTER        ?= cluster
CLUSTER_ROOT   ?= /home/$(USER)/aneurysm_cnn/training_engine/experiments
LOCAL_RESULTS  ?= ./results

# Find the most-recently-modified experiment directory on the cluster and print its name.
.PHONY: cluster-latest
cluster-latest:
	ssh $(CLUSTER) "ls -td $(CLUSTER_ROOT)/*/ | head -1"

# Sync logs and checkpoints for a specific experiment back to ./results/.
# Usage: make cluster-fetch EXP=R3D50_A192_lr1e4
.PHONY: cluster-fetch
cluster-fetch:
	@test -n "$(EXP)" || (echo "Usage: make cluster-fetch EXP=<experiment_name>"; exit 1)
	mkdir -p $(LOCAL_RESULTS)/$(EXP)
	rsync -avz --progress \
	    --include="*/" \
	    --include="*.json" \
	    --include="*.csv" \
	    --include="*.png" \
	    --include="*_model_best.pth" \
	    --exclude="*_model_last.pth" \
	    --exclude="*.pth" \
	    $(CLUSTER):$(CLUSTER_ROOT)/$(EXP)/ \
	    $(LOCAL_RESULTS)/$(EXP)/

# Sync all experiment directories (logs, CSVs, PNGs, best checkpoints only).
# Large *_model_last.pth files are excluded to keep transfer times short.
.PHONY: cluster-fetch-all
cluster-fetch-all:
	mkdir -p $(LOCAL_RESULTS)
	rsync -avz --progress \
	    --include="*/" \
	    --include="*.json" \
	    --include="*.csv" \
	    --include="*.png" \
	    --include="*_model_best.pth" \
	    --exclude="*_model_last.pth" \
	    --exclude="*.pth" \
	    $(CLUSTER):$(CLUSTER_ROOT)/ \
	    $(LOCAL_RESULTS)/
```

**What is synced and what is left on the cluster:**

| Included (synced locally) | Excluded (stays on cluster) |
|---|---|
| `best_checkpoint_metadata.json` | `*_model_last.pth` (final-epoch weights, typically 100–500 MB each) |
| `*_evaluation_summary.csv` | `*.pyc`, MONAI cache dirs |
| `*_metrics.csv` | |
| `*_predictions.csv` | |
| `*.png` (confusion matrices, ROC curves) | |
| `*_model_best.pth` (AUC-optimal checkpoint) | |

The `--include`/`--exclude` order in rsync is evaluated top-to-bottom; the `--exclude="*.pth"` catch-all at the bottom prevents any other `.pth` files that match no `--include` pattern from being transferred. `*_model_best.pth` is included explicitly so the best checkpoint is always available locally for inference or fine-tuning.

**Finding the latest output directory without rsync:**

```bash
ssh cluster "ls -td ~/aneurysm_cnn/training_engine/experiments/*/ | head -1"
```

---

### 3. Library consolidation

**Third-party libraries used across `src/` and `train_models.py`:**

| Library | Used in | Purpose |
|---|---|---|
| `torch` | all modules | tensors, models, optimisers, AMP, DataLoader |
| `monai` | `data_preprocess.py`, `models.py` | transforms, CacheDataset, ResNet/DenseNet/SwinUNETR/UNETR architectures |
| `torchvision` | `models.py` | R3D-18 video backbone |
| `numpy` | `data_preprocess.py`, `plots.py` | array ops, `bincount`, confusion matrix normalisation |
| `scikit-learn` | `data_preprocess.py`, `training.py`, `plots.py` | `StratifiedKFold`, `train_test_split`, `fbeta_score`, `roc_auc_score`, `roc_curve`, `auc`, `confusion_matrix`, `precision_score`, `recall_score` |
| `matplotlib` | `plots.py` | confusion matrix and ROC curve PNGs |
| `pandas` | `plots.py` | DataFrame construction for CSV writing and metric aggregation |

**Stdlib modules used:**
`os`, `json`, `csv`, `random`, `copy`, `time`, `shutil`, `argparse`, `typing`, `itertools`

**Redundancies and replacement opportunities:**

| Redundancy | Detail | Remedy |
|---|---|---|
| `pandas` used only for CSV output | `plots.py` uses `pd.DataFrame` and `.to_csv()` solely to write CSVs and compute `mean()`/`std()` over a handful of floats | Replace with `csv.DictWriter` (already imported in the same file) for CSV writing and inline arithmetic for mean/std — eliminating the `pandas` dependency entirely from `plots.py`. The only reason to keep `pandas` is if richer downstream analysis is planned. |
| `csv` module imported alongside `pandas` | `plots.py` imports both `csv` (for `save_predictions_from_detailed_results`) and `pandas` (for `evaluate_models` and `save_metrics_to_csv`) | If `pandas` is retained, `csv.DictWriter` can be replaced by `pd.DataFrame(...).to_csv()`; if `pandas` is dropped, all CSV writing unifies under the stdlib `csv` module. |
| `sklearn.metrics.auc` + `roc_curve` in `plots.py` vs `roc_auc_score` in `training.py` | Both compute ROC-AUC, but `plots.py` re-runs inference to collect probabilities while `training.py` collects them during the validation pass | `plot_roc_curve` could accept pre-collected `(fpr, tpr)` or `(labels, probs)` arrays rather than a live DataLoader, removing the duplicate inference pass and the need for a live model at artifact-save time. The `auc(fpr, tpr)` call in `plots.py` would then match what `roc_auc_score` already computes in `training.py`. |
| `itertools.product` in `plots.py` | Used for a 2×2 grid iteration in `plot_confusion_matrix`; could be written as a nested `for i in range(...): for j in range(...)` | Not a real redundancy — `itertools` is stdlib and the usage is idiomatic. No change needed. |

---

### 4. Modularity flags

The most significant coupling issues that make it hard to swap a model or change training strategy without touching multiple files:

---

#### 4.1 `plots.py::plot_roc_curve` takes a live model and DataLoader

**Problem.** `plot_roc_curve` accepts `model: nn.Module` and `dataloader: DataLoader`, re-runs a full forward pass to collect probabilities, and then plots. This means artifact generation is coupled to an active model on a device. The function also hardcodes the `"image"` and `"label"` batch dict keys; any DataLoader that yields differently-named keys silently produces wrong results (the label tensor will be missed and the function will likely raise a `KeyError` or produce garbage AUC).

**Proposed fix.** Change the signature to accept pre-collected arrays:

```python
def plot_roc_curve(
    labels: List[int],
    probs: List[float],
    filename: str,
) -> None:
```

The caller (`save_fold_artifacts`) already has `detailed_results` from `validate_one_epoch(..., return_details=True)`. Probabilities are currently not stored in `detailed_results`, so add `"prob"` to each per-sample dict in `validate_one_epoch`. This is a one-line addition to the existing dict append. With this change, `save_fold_artifacts` no longer needs `model` or `eval_loader` at all — the artifacts function becomes pure data-in / file-out with no device dependency.

---

#### 4.2 `orchestrator.py` imports directly from every module

**Problem.** `run_experiment` imports and directly calls functions from all four `src` submodules (`data_preprocess`, `models`, `training`, `plots`). Adding a new training strategy (e.g. a learning-rate scheduler, a different optimiser, or a custom loss) requires editing `run_one_fold` in `orchestrator.py` and potentially `training.py`. There is no interface contract between the orchestrator and the training module; the coupling is direct function calls.

**Proposed minimal fix.** The training strategy is the most likely axis of change. Extract the optimiser and scheduler setup from `run_one_fold` into a small factory:

```python
# training.py — add this function
def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config.get("WEIGHT_DECAY", 1e-4),
    )
```

Then `run_one_fold` calls `training.build_optimizer(model, config)` instead of constructing `AdamW` directly. This is the minimum needed so that changing from AdamW to SGD (or adding a scheduler) is a single-file change in `training.py` rather than requiring an edit to the orchestrator.

---

#### 4.3 `models.py` has three separate locations that must be updated in sync

**Problem.** Adding a new model requires three coordinated edits: (1) add a string to `SUPPORTED_MODELS`, (2) add a branch to the `if/elif` chain in `get_model`, and (3) add a branch to `_strip_classifier` (if the model uses tabular fusion). These sites are not co-located and there is no compile-time or test-time check that they are in sync.

**Proposed fix.** Replace the three separate locations with a single model registry dict:

```python
# models.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class _ModelEntry:
    build: Callable[..., nn.Module]       # constructor
    classifier_attr: str                   # attribute path to the final linear layer

_REGISTRY: Dict[str, _ModelEntry] = {
    "R3D18":        _ModelEntry(get_r3d18_pytorch_model,  "fc"),
    "R3D18_MONAI":  _ModelEntry(get_r3d18_monai_model,    "fc"),
    "R3D10":        _ModelEntry(get_r3d10_model,          "fc"),
    "R3D50":        _ModelEntry(get_r3d50_model,          "fc"),
    "DenseNet121":  _ModelEntry(get_densenet121_monai_model, "class_layers.out"),
    # ...
}

SUPPORTED_MODELS = list(_REGISTRY.keys())   # derived, not maintained separately

def get_model(model_name, num_classes, use_tabular=False, spatial_size=(128,128,128)):
    entry = _REGISTRY.get(model_name.upper())
    if entry is None:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {SUPPORTED_MODELS}")
    model = entry.build(num_classes=num_classes)
    if use_tabular:
        feat_dim = _strip_classifier_by_attr(model, entry.classifier_attr)
        model = MultiModalWrapper(model, feat_dim, tabular_dim=2, num_classes=num_classes)
    return model
```

Adding a new model is now a single dict entry. The `_strip_classifier` `if/elif` chain is replaced by `_strip_classifier_by_attr(model, attr_path)` which uses `attrgetter` or a simple dotted-path walk. This is the worst offender in terms of maintenance surface and the fix is self-contained within `models.py`.

---

#### 4.4 Config dict accessed by string literals throughout the orchestrator

**Problem.** `run_experiment` and `run_one_fold` access config keys as bare string literals (`config["LEARNING_RATE"]`, `config.get("GRAD_ACCUM_STEPS", 1)`, etc.) spread across ~15 call sites. A key rename requires grep-and-replace across multiple files with no static check.

**Proposed minimal fix.** Define a `Config` dataclass or `TypedDict` in `train_models.py` and use it as the type annotation throughout. No runtime behaviour changes; mypy or pyright immediately flags any key mismatch. This is a zero-risk structural change that makes the implicit schema explicit without touching any logic.

```python
from typing import TypedDict

class ExperimentConfig(TypedDict, total=False):
    name: str
    model: str
    LEARNING_RATE: float
    EPOCHS: int
    # ... all keys from DEFAULT_CONFIG
```

The orchestrator signature becomes `run_experiment(config: ExperimentConfig)` and static analysis tools will flag any key spelling error.
