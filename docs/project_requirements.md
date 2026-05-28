# Project Requirements

## 1. Data Quality

### 1.1 DICOM Ingestion

- Accept a raw folder of per-case DICOM subdirectories; traverse and rename folders to a canonical `BP{NNN}` scheme.
- Validate each series: check geometric consistency (uniform slice spacing, no gaps), orientation (`get_orientation` → axial/coronal/sagittal), modality presence, and slice-thickness uniformity.
- Detect and flag mixed-series folders (multiple UIDs in one folder); report per-series breakdown in `mixed_series_analysis.csv` (this file stays separate — it is per-series, not per-case).
- Filter scout slices (`ImageType` must contain `'ORIGINAL'`); record `scout_slice_count` per case.
- Flag geometric outliers by exam size (`_flag_exam_size_outliers`); include `exam_size` in the per-case report.
- Attach ground-truth label (`class`, `location`) from `classes.csv` via suffix-stripped exam-name lookup; mark unlabelled valid cases as `MISSING_CLASS`.
- Produce a single **`pipeline_report.csv`** (one row per input case, 19 columns) that merges all per-case fields currently split across `folder_rename_map.csv` and `ingestion_summary.csv`:
  - `original_name`, `fixed_name`, `data_path`, `total_dcms`
  - `issues` (pipe-separated list of all flags, e.g. `VARIABLE_SPACING|MISSING_CLASS`), `duplicate_slice_count`, `scout_slice_count`
  - `orientation`, `modality`, `slice_thickness`, `patient_age`, `patient_sex`
  - `image_dimensions`, `class`, `location`, `exam_size`
  - `conversion_status`, `conversion_reason`, `output_path`
- All issues for a case must be accumulated, not overwritten. If a case has multiple problems (e.g. `VARIABLE_SPACING` and `MISSING_CLASS`), all must appear in the `issues` field; no issue may silently replace another.
- Cases that never reached conversion must appear in `pipeline_report.csv` with `conversion_status = "not_attempted"` and `conversion_reason` derived from the `issues` field.
- `folder_rename_map.csv` and `ingestion_summary.csv` are retired once `pipeline_report.csv` is in place.

### 1.2 Diagnostics Consolidation

Current diagnostics scripts (9 total) and their disposition:

| Script | Status | Merged subcommand |
|---|---|---|
| `survey_nifti.py` | Merge | `diagnose.py survey` |
| `analyse_nifti_survey.py` | Merge | `diagnose.py analyse` |
| `check_nifti.py` | Merge | `diagnose.py check FILE` |
| `check_nifti_quality.py` | Merge | `diagnose.py quality FILE` |
| `check_resize_quality.py` | Merge | `diagnose.py resize` |
| `check_fov_air.py` | Merge | `diagnose.py fov-air` |
| `compute_optimal_spacing.py` | Merge | `diagnose.py spacing` |
| `extract_gender.py` | **Delete** — superseded by `inventory_scan.write_gender_to_classes_csv` |
| `check_missing_gender.py` | **Delete** — superseded by `inventory_scan._log_status_breakdown` |

- The 7 surviving scripts merge into `data_engine/diagnostics/diagnose.py` under an `argparse` subcommand interface; all share `nibabel`/`numpy` and operate on the same `nifti/` output directory.
- `diagnose.py survey` runs first; `analyse`, `resize`, `fov-air`, and `spacing` depend on its `nifti_survey.csv` output.
- All diagnostic output goes to `data_engine/diagnostics/outputs/`; no diagnostic script writes to the main pipeline CSVs.
- Deleting `check_missing_gender.py` removes the only `pandas` dependency in the diagnostics directory.
- **Artifact budget: the entire data quality step must produce at most 3 output files:** `pipeline_report.csv` (per-case ingestion + conversion), `diagnostics_report.csv` (all post-conversion checks merged into one row per case), and `mixed_series_analysis.csv` (stays separate — it is per-series, not per-case). No other CSVs, no JSONL logs.
- See `docs/de_diagnostics.md` for per-script detail.

### 1.3 NIfTI Conversion

- Convert validated DICOM series to NIfTI (`.nii.gz`) using ITKReader; preserve the affine matrix; cast to float32.
- Write NIfTI files to `--nifti-dir`; guard against writes to local disk when the target filesystem is unmounted.
- After successful NIfTI conversion, discard raw DICOM source data to recover disk space (do not store both).
- Record `conversion_status` (`success` / `skipped` / `failed`) and `output_path` per case in `pipeline_report.csv`.
- Only cases with `validation_status == 'OK'` and a valid class label proceed to conversion (`filter_for_conversion`).

### 1.4 Incremental Ingestion

- Support a `--incremental` (or equivalent) mode that:
  - Reads the existing `nifti/` directory to build a set of already-converted case names.
  - Scans a secondary raw input folder (e.g. `raw2/`) for new cases not present in that set.
  - Runs the full ingestion pipeline only on the new cases; skips all existing ones.
  - Appends new rows to `pipeline_report.csv` rather than rewriting it.
- No re-processing of existing NIfTI files; no modification of already-converted data.

### 1.5 Cases Grouping

- `diagnose.py survey` must identify all unique acquisition configurations (distinct `(dim_x, dim_y, dim_z, spacing_x, spacing_y, spacing_z)` tuples) present in the NIfTI dataset and cluster them into scanner-protocol groups using a greedy tolerance-based algorithm (XY matrix match + 15% spacing_x / 20% spacing_z thresholds).
- Groups are the primary unit for dataset-level analysis: FOV coverage, air depth, and resize quality checks all operate per-group.
- The survey output must record `group_id` and `group_count` per case so any downstream script can filter or stratify by scanner group without re-running the survey.
- See `docs/data_engine/de_diagnostics.md` → `survey_nifti.py` for the current grouping algorithm.

## 2. Dataset Generation

### 2.1 Dataset Approaches

- Six preprocessing variants (A–F) must be supported, forming a 3×2 factorial design:
  - **Crop axis:** 80% centre-crop (A, C, E) vs. full volume (B, D, F).
  - **Resampling strategy:** no resampling / trilinear resize (C, D); per-scan dynamic isotropic spacing (A, B); fixed dataset-wide isotropic spacing (E, F).
- Three output resolutions must be supported: `128×128×128`, `192×192×128`, `256×256×176`; each variant can be generated at any resolution independently.
- Input files must be sourced from the NIfTI output of `data_cleaner.py`, organised into `0/` and `1/` class subdirectories.
- Output directory layout mirrors input class structure; filenames are preserved; one output directory per (variant, resolution) combination.
- Fixed-spacing variants (E, F) derive their spacing from dataset-wide FOV maxima (192 mm for cropped, 240 mm for full); these constants must be updated if the dataset grows to include larger FOVs.

### 2.2 Local vs. Cluster

- Full datasets (all variants, all cases) are generated exclusively on the cluster via SLURM (`dataset_gen.slurm` / `dataset_gen_128.slurm`).
- Local disk holds only small sample datasets (a subset of cases) for development and smoke-testing; sample datasets use the same variant pipeline with the same parameters.
- Sample datasets must include at least one case from each scanner-protocol group (see §1.5) so all acquisition types are exercised during local testing.
- Sample datasets must be reproducible from the same NIfTI input files and the same CLI arguments.

### 2.3 Dataset Versioning and Traceability

- Each generated dataset directory must be traceable to: the set of input NIfTI files that produced it, the variant key, the target resolution, and the `--seed` used for file-list shuffling.
- If a dataset directory is regenerated with different parameters, the old one must not be silently overwritten; versioning or namespacing by parameters is required.
- Processing failures (per-file errors) must be logged and surfaced as a non-zero exit code; no silent partial datasets.

### 2.4 Reproducibility

- The file-list shuffle seed (`--seed`, default `42`) must be recorded alongside the dataset so any run can be reproduced exactly.
- Worker-level processing is deterministic given the same input file and variant; no random transforms are applied during dataset generation.
- Idempotency: re-running `dataset_gen.py` on a completed dataset skips already-written files without reprocessing them.

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

## 5. Infrastructure

### 5.1 Environment Management

- Local (`aneurysm_cnn_data`) and HPC (`aneurysm_cnn_data_hpc`) conda environments must pin Python at the same granularity (patch-level `3.11.15`, not minor-only `3.11`).
- `pydicom`, `nilearn`, and `itk` are present locally but absent on HPC; if any data-engine script importing these is expected to run on HPC, the HPC env must be updated to include them.
- `pytest` and `ruff` are intentionally local-only; their absence on HPC is acceptable and need not be reconciled.
- The `setuptools` pin comment must state the same reason in both YAMLs (currently mismatched: local says `itk`, HPC says `monai`).
- The `torch` CPU-build entry on HPC (required by MONAI transforms) must be explicitly noted in a comment in the local env file as intentionally absent locally.

### 5.2 SLURM Scripts

- All SLURM scripts must use the correct partition for their workload: CPU-only jobs on `cpu`, GPU jobs on `gpu-amd`.
- **Known bug:** `dataset_gen_128.slurm` requests partition `gpu-amd` for a CPU-only workload — must be changed to `cpu`.
- `start_script_rocm.slurm` must include an explicit `--mem` directive; defaulting to a per-CPU allocation is insufficient for large 3D volume training.
- All scripts must have a meaningful `--job-name`; `teste_sbatch` in `test_rocm.slurm` is a stale placeholder and must be replaced.
- Wall times must reflect actual job duration: the 6-hour wall time in `test_rocm.slurm` (a seconds-long smoke test) must be reduced.
- Stdout and stderr handling must be consistent; diagnostic jobs should merge both into a single timestamped log.
- Conda activation must be consistent across scripts; hardcoding `$HOME/miniconda3` is fragile — use the multi-path fallback pattern from `diagnose_rocm.slurm`.
- `dataset_gen_128.slurm` must make `VARIANTS` configurable via env var, matching `dataset_gen.slurm`.
- `dataset_gen_128.slurm` must add a version verification block (monai/nibabel/numpy) for reproducibility parity.

### 5.3 ROCm / Environment Diagnostics

- The three current Python diagnostic scripts (`check_kfd_version.py`, `diagnose_rocm.py`, `test_rocm.py`) must be merged into a single `check_environment.py` with a `--sections` CLI (`--kfd`, `--disk`, `--env`, `--torch`, `--all`).
- `--all` must reproduce the full combined output of all three current scripts with no loss of coverage.
- The KFD ioctl + HSA library `strings` scan from `check_kfd_version.py` must be retained as it is unique.
- The merged script must be runnable both locally (without a GPU) and via a single SLURM wrapper replacing the current two wrappers.

### 5.4 Data Transfer

- `transfer_nifti.sh` must replace `scp -r` with `rsync -aP` to make transfers resumable.
- Both transfer scripts must include a zero-file guard (exit with error if no local files match the pattern) — currently missing from `transfer_nifti.sh`.
- Both scripts must perform post-transfer file-count verification and exit non-zero on mismatch.
- Once both scripts use identical `rsync` logic, they may be unified under a single script with a `--mode nifti|zips` flag.
- SSH keepalive options (`ServerAliveInterval`, `ServerAliveCountMax`) and the cluster alias must be defined in one shared location (e.g. `~/.ssh/config`) rather than duplicated across scripts.

### 5.5 Cluster Workflow

- The standard deploy cycle (local edit → `ssh drummond` → `git pull` → `sbatch`) must be documented and optionally wrapped in a Makefile target.
- A `make fetch` target (or equivalent) must pull results from the cluster using `rsync`; see section 4.3 for sync filter requirements.
- SLURM `.out` log files must be written to (or copied into) the relevant experiment output directory, not left in the submission directory.
- A `make status` or equivalent must show running/pending jobs for the project without manual `squeue` invocations.
