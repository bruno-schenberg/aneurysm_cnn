# Training Engine Requirements

## 3. Model Training

### 3.1 Existing Features

- **Architectures available:** R3D10, R3D18 (torchvision Kinetics pretrained), R3D18_MONAI, R3D50, DenseNet121, SwinUNETR, UNETR, UNet3D, UNet3DWithBackbone, MIL_R3D18 — all sharing the same `(B, 1, D, H, W)` → logits `(B, 2)` interface.
- **Multimodal fusion:** `MultiModalWrapper` fuses image backbone features with tabular input (age, gender) via late concatenation; tabular branch is a 2-layer MLP.
- **Training loop:** per-epoch train and validation passes with gradient accumulation, mixed-precision (`bfloat16` AMP), and early stopping (patience-based, AUC-triggered).
- **Checkpointing:** best model selected by `val_AUC` (threshold-independent); checkpoint kept in memory during training, saved to `model_best.pth` after fold completes; `model_last.pth` also written.
- **Metrics tracked per epoch:** `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_F2`, `val_AUC`.
- **Metrics reported at eval:** accuracy, F2 (β=2, recall-weighted for clinical asymmetry), ROC-AUC, confusion matrix; per-sample predictions CSV; fold-level and experiment-level summary CSVs.
- **Class imbalance handling:** two strategies selectable per experiment — `weighted_cost_function` (inverse-frequency loss weights) or `oversampling` (`WeightedRandomSampler`).
- **Cross-validation:** stratified k-fold or single train/val/test split; held-out test set carved out before any fold logic; class ratio preserved across all partitions.
- **Augmentation (training only):** 3-axis random flip, discrete 90° rotation, continuous ±15° rotation, intensity scaling ±10%, random zoom 90–110%, Gaussian noise; all applied via MONAI dictionary transforms.
- **Reproducibility:** global seed applied to Python, NumPy, PyTorch, MONAI, and per-DataLoader worker RNGs before any split or augmentation.
- **Experiment config system:** flat JSON list of experiment dicts merged with `DEFAULT_CONFIG` defaults; single `--experiments` CLI flag; per-experiment exception isolation so one failure does not abort the full run.
- **Artifact layout:** `experiments/{name}/fold_{n}/` per fold, plus `{name}_evaluation_summary.csv` at experiment root; optional backup via `EXPERIMENT_BACKUP_DIR` env var.

### 3.2 Missing / Needed

- **Probability output missing from `detailed_results`:** `validate_one_epoch` records `pred_label` but not the softmax probability; `plot_roc_curve` compensates by re-running a full forward pass over the DataLoader — a redundant inference pass that couples artifact generation to a live model on a device.
- **Augmentation not configurable from config:** all probabilities, rotation ranges, zoom bounds, and noise std are hardcoded in `get_transforms`; changing them requires code edits, not config changes.
- **`num_workers` hardcoded:** DataLoader worker counts (4 train / 2 val/test) are fixed constants rather than config or system-derived values.
- **No learning-rate scheduler support:** `AdamW` is hardcoded in the orchestrator with no scheduler; adding cosine decay or warmup requires editing `run_one_fold` directly.
- **No weight-decay config key:** `weight_decay` for `AdamW` is hardcoded to `1e-4`; not exposed in the experiment config.
- **`SwinUNETRClassifier` weight path is hardcoded:** pretrained weights loaded from an absolute `/root/.cache/...` path; silently falls back to random init on any other system with no runtime indication.
- **`MILClassifier` batch size limited to 1 silently:** batch size > 1 causes a shape error at runtime; no guard or clear error message.
- **No precision/recall/specificity in per-fold summary:** `evaluation_summary.csv` aggregates AUC and F2 across folds but does not include precision, recall, or specificity, which are standard for clinical classification tasks.
- **No test-time inference script:** no standalone script to run a saved `model_best.pth` on new data outside the cross-validation loop.
- **Model left in last-epoch state after `run_training_loop`:** caller must manually call `model.load_state_dict(best_checkpoint["state_dict"])`; this implicit contract is undocumented and easy to miss.
- **`models.py` requires three-site update per new architecture:** `SUPPORTED_MODELS`, `get_model` dispatch chain, and `_strip_classifier` must all be updated in sync with no static enforcement.

### 3.3 Experiment Management

- **Current state:** five flat `experiments*.json` files at `training_engine/` root; every entry repeats 7–11 identical fields; changing one shared hyperparameter (e.g. `EARLY_STOPPING_PATIENCE`) requires editing every entry individually.
- **Required:** a `configs/` subdirectory with base config files (one per resolution/theme) and experiment list files that reference a base and list only per-experiment overrides; loader must support both flat-list format (backward-compatible) and the new `{"base": "...", "experiments": [...]}` format.
- **Required:** experiment config files named by convention: `base_{resolution}.json`, `{model}_{resolution}_{theme}.json`.
- **Required:** a way to list, filter, and rerun specific experiments by name without editing the JSON file (e.g. CLI flag `--filter` or a `smoke_test` flag per entry).
- **Required:** experiment output directory names must match the `name` field exactly and be stable across reruns; no timestamp suffixes that prevent comparison between runs.

## 4. Output Generation

### 4.1 Existing Outputs

Per fold, written to `experiments/{name}/fold_{n}/`:
- `{name}_fold{n}_cm_raw.png` — unnormalised confusion matrix
- `{name}_fold{n}_cm_normalized.png` — row-normalised confusion matrix
- `{name}_fold{n}_roc_curve.png` — ROC curve (generated via re-inference, not from cached probs)
- `{name}_fold{n}_predictions.csv` — per-sample `filename`, `true_label`, `prediction`
- `{name}_fold{n}_metrics.csv` — per-epoch `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_F2`, `val_AUC` + final-eval row
- `{name}_fold{n}_model_last.pth` — weights at final epoch
- `{name}_fold{n}_model_best.pth` — weights at best `val_AUC` epoch
- `best_checkpoint_metadata.json` — `best_epoch`, `best_val_auc`, `best_val_f2`

Per experiment, written to `experiments/{name}/`:
- `{name}_evaluation_summary.csv` — per-fold rows + `Average` and `Std. Dev.` footer; columns: Precision, Recall, F2-Score, Eval Accuracy, Eval Loss

Optional:
- Full experiment directory copied to `$EXPERIMENT_BACKUP_DIR` via `shutil.copytree` if env var is set

### 4.2 Missing / Needed

- **Softmax probabilities not saved:** `predictions.csv` records only hard labels; no probability column; ROC curve compensates with a redundant forward pass.
- **AUC absent from `evaluation_summary.csv`:** the per-epoch AUC is tracked for checkpointing but never appears in the cross-fold summary CSV.
- **Precision, recall, specificity absent from summary:** `evaluation_summary.csv` aggregates F2 and accuracy but omits precision, recall, and specificity — standard for clinical classification reporting.
- **No standalone inference script:** no way to run `model_best.pth` on new NIfTI files outside the cross-validation loop; prediction on held-out or prospective cases requires a new script.
- **No exportable model format:** checkpoints are raw PyTorch state dicts; no ONNX or TorchScript export for deployment or cross-framework use.
- **No per-experiment human-readable summary:** no single text or JSON file that captures key results (best fold AUC, mean AUC ± std, config snapshot) for quick review without opening CSVs.
- **SLURM job log not linked to experiment output:** `.out` files from `sbatch` are written to the submission directory, not to `experiments/{name}/`; correlating a log with a run requires manual filename matching.

### 4.3 Result Retrieval from Cluster

- **Required:** an `rsync`-based fetch command (Makefile target or shell script) to pull results from the cluster to local disk.
- Sync selectively: include `*.json`, `*.csv`, `*.png`, `*_model_best.pth`; exclude `*_model_last.pth` (large, rarely needed locally).
- Support fetching a single named experiment (`EXP=<name>`) or all experiments in one pass.
- Cluster root path must be configurable (env var or Makefile variable); default derived from known cluster layout (`~/aneurysm_cnn/training_engine/experiments`).
- A `cluster-latest` convenience target should print the most recently modified experiment directory on the cluster without fetching it.
