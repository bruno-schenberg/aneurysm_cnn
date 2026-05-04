# 3D CNN for Aneurysm Detection

A masters thesis project implementing a full end-to-end pipeline for training and evaluating 3D Convolutional Neural Networks (CNNs) for the detection of intracranial aneurysms in volumetric medical imaging data.

The system is split into two independent stages: a **data engine** that converts raw DICOM scans into standardized NIfTI volumes and generates multiple preprocessing variants, and a **training engine** that trains, cross-validates, and evaluates 3D CNN models across configurable experiments.

---

## Pipeline Overview

```
Raw DICOM files
      │
      ▼
┌─────────────────────────────────┐
│         Stage 1: Data Engine    │
│                                 │
│  1. Folder discovery & naming   │
│  2. DICOM series validation     │
│  3. Mixed series analysis       │
│  4. Class label joining         │
│  5. NIfTI conversion            │
│  6. Dataset variant generation  │
└─────────────┬───────────────────┘
              │  /mnt/data/cases-3/
              │  dataset_{A,B,C,D,E}/
              │       0/  ← healthy
              │       1/  ← aneurysm
              ▼
┌─────────────────────────────────┐
│       Stage 2: Training Engine  │
│                                 │
│  1. NIfTI file discovery        │
│  2. Hold-out test split         │
│  3. Stratified k-fold split     │
│  4. MONAI transforms            │
│  5. DataLoader assembly         │
│  6. Model initialisation        │
│  7. Training loop               │
│  8. Validation + F2 checkpoint  │
│  9. Final evaluation            │
│ 10. Artifact generation         │
└─────────────────────────────────┘
```

---

## Core Libraries

| Library | Role |
|---------|------|
| **PyTorch** | Deep learning framework; training loop, optimiser, loss |
| **MONAI** | Medical imaging transforms, model architectures (ResNet, DenseNet, SwinUNETR), DataLoader |
| **torchvision** | R3D-18 video model with Kinetics-400 pretrained weights |
| **NiBabel** | Reading and writing NIfTI (`.nii.gz`) volumetric files |
| **pydicom** | Parsing and validating raw DICOM files |
| **scikit-learn** | Stratified k-fold splitting, F-beta score, ROC curve metrics |
| **NumPy** | Array operations used throughout image processing |
| **pandas / matplotlib** | Tabular result aggregation and plot generation |

---

## Environment Setup

The project uses Python virtual environments (`.venv`) for local development, and Conda environments for execution on the High-Performance Computing (HPC) cluster. Environment configurations are located in the `infrastructure/` directory.

| Target | Engine | Type | Path |
|--------|--------|------|------|
| Local | Data | `.venv` | Manual setup via `pip` (see packages in `infrastructure/data/environment_data.yml`) |
| Local | Training (CUDA) | `.venv` | Manual setup via `pip` (see packages in `infrastructure/training/environment_cuda.yml`) |
| HPC | Data | Conda | `infrastructure/data/environment_data_hpc.yml` |
| HPC | Training (ROCm) | Conda | `infrastructure/training/environment_rocm.yaml` |

### Local Setup (.venv)

For local development, we use two separate standard Python virtual environments, located directly inside each engine's parent folder.

```bash
# Data Engine (.venv)
cd data_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../infrastructure/data/environment_data.yml # Equivalent pip packages
deactivate
cd ..

# Training Engine with CUDA (.venv)
cd training_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../infrastructure/training/environment_local.yml # Equivalent pip packages
deactivate
cd ..
```

### HPC Setup (Conda)

On the HPC cluster, conda environments are used to manage complex dependencies:

```bash
# Activate conda if not already available in your shell
source /opt/miniconda3/etc/profile.d/conda.sh

# Create the data engine environment
conda env create -f infrastructure/data/environment_data_hpc.yml
conda activate aneurysm_cnn_data

# Create the training environment for ROCm GPU
conda env create -f infrastructure/training/environment_rocm.yaml
conda activate aneurysm_cnn_rocm
```

---

## Stage 1: Data Engine

The data engine lives in `data_engine/` and runs exclusively on the local development machine. Its job is to turn a directory of raw, inconsistently named DICOM case folders into clean NIfTI volumes organized by class label, and then generate five distinct preprocessing variants for experimentation.

Entry point: `data_engine/data_cleaner.py`

```bash
source .venv_data/bin/activate
python data_engine/data_cleaner.py \
    --raw-dir /mnt/data/cases-3/raw \
    --nifti-dir /mnt/data/cases-3/nifti
```

### Step 1 — Folder Discovery and Name Standardization (`file_utils.py`)

Raw DICOM cases arrive in folders with inconsistent names (e.g. spaces, mixed case, typos). `get_subfolders` lists all immediate subdirectories of `--raw-dir`, and `organize_data` normalizes each folder name to the canonical format `BP001`, `BP002`, etc. It records both the original and the fixed name so the change is fully traceable in the output CSV.

This step produces no files; it populates an in-memory list of case records that flows through the rest of the pipeline.

### Step 2 — DICOM Series Validation (`dicom_utils.py`)

`validate_dcms` inspects every DICOM file in each case folder. For each series it checks:

- **Duplicate slices** — identifies and counts slices with identical `InstanceNumber` metadata, which typically indicates a corrupted or re-exported study.
- **Scout images** — detects localizer scout scans (low-slice-count series used for scan planning) that must be excluded from training.
- **Spatial consistency** — verifies that all slices in a series share the same `Rows`, `Columns`, and `SliceThickness`. Inconsistent spatial metadata means the series cannot be stacked into a coherent 3D volume.
- **Orientation** — records the dominant image orientation (axial, coronal, sagittal).
- **Modality** — records the DICOM modality (CT, MR, etc.) and flags unexpected values.

Each case receives a `validation_status` of `OK`, `WARN`, or `FAIL`. Failed cases are excluded from conversion.

### Step 3 — Mixed Series Analysis (`dicom_utils.py`)

Some case folders contain more than one series (e.g. a contrast and a non-contrast acquisition). `analyze_mixed_folders` produces a CSV report (`mixed_series_analysis.csv`) listing every series in those folders with its slice count, orientation, and modality. This report is used for manual review — decisions about which series to keep cannot be made automatically and are documented here for audit purposes.

### Step 4 — Class Label Joining (`class_utils.py`)

`join_class_data` reads `dataset/classes.csv`, the ground-truth label file checked into the repository. It joins each validated case to its label (`0` = healthy, `1` = aneurysm) on the standardized case name. `check_missing_class` then flags any case that passed validation but has no label — those cases cannot be used for supervised training and are excluded.

`classes.csv` is the only file in the repository that encodes clinical ground truth. It is never generated by code; it was created manually from clinical records.

### Step 5 — NIfTI Conversion (`nifti_utils.py`)

`filter_for_conversion` takes the final case list and keeps only cases with status `OK` and a valid class label.

`process_and_convert_exams` converts each eligible case in parallel using a configurable worker pool (`--workers`, default: `os.cpu_count()`). For each case it uses MONAI and ITK to:

1. Load the DICOM series from disk.
2. Stack slices into a 3D volume respecting the `ImagePositionPatient` and `ImageOrientationPatient` DICOM tags.
3. Save the result as a compressed NIfTI file (`.nii.gz`) into a class subdirectory:
   - `/mnt/data/cases-3/nifti/0/` ← healthy cases
   - `/mnt/data/cases-3/nifti/1/` ← aneurysm cases

The class subdirectory layout is used directly by the training engine's data loader.

After conversion, `_build_audit_log` produces a complete audit CSV (`ingestion_summary.csv`) covering every input case — including those rejected at earlier stages — so the conversion run is fully traceable. A validation summary CSV (`folder_rename_map.csv`) is also written with per-case metadata.

### Step 6 — Dataset Variant Generation (`dataset_gen.py`, `nifti_resize.py`)

The training engine requires all volumes to be the same size (128×128×128 voxels). There are multiple valid ways to get there from volumes with different native resolutions and fields of view. Rather than committing to one, `dataset_gen.py` generates five preprocessing variants in parallel so that experiments can compare them.

```bash
python data_engine/dataset_gen.py --workers 3
```

`dataset_gen.py` discovers all `.nii.gz` files in the `nifti/` output tree and dispatches each file to a pool of worker processes. Each worker calls all five variant functions from `nifti_resize.py` and mirrors the `0/` / `1/` subdirectory structure into each output directory.

| Key | Output Path | Description |
|-----|-------------|-------------|
| A | `dataset_A_resampled_cropped` | Resample to 1 mm isotropic spacing, then centre-crop or pad to 128³ |
| B | `dataset_B_resampled_shrunk` | Resample to 1 mm isotropic spacing, then shrink (resize) to 128³ |
| C | `dataset_C_cropped` | Native voxel spacing preserved, centre-crop or pad to 128³ |
| D | `dataset_D_shrunk` | Native voxel spacing preserved, shrink (resize) to 128³ |
| E | `dataset_E_isotropic_padded` | Resample largest dimension to 128 px (preserving aspect ratio), then zero-pad to 128³ |

**Variant A and B** normalize voxel spacing to 1 mm isotropic before resizing. This makes physical measurements comparable across scanners with different slice thicknesses, at the cost of resampling artefacts.

**Variant C and D** preserve native spacing. This avoids resampling but means the same physical region may cover different numbers of voxels depending on the scanner.

**Variant E** preserves the aspect ratio by scaling only the longest dimension, then pads the shorter dimensions with zeros. This avoids distorting the anatomy.

Resampling uses MONAI's `Spacing` transform (bilinear interpolation). Cropping/padding use MONAI's `ResizeWithPadOrCrop`. Shrinking uses MONAI's `Resize`.

---

## Stage 2: Training Engine

The training engine lives in `training_engine/` and handles experiment configuration, data loading, model training, cross-validation, and artifact generation.

Entry point: `training_engine/train_models.py`

```bash
source .venv_cuda/bin/activate   # or aneurysm_cnn_rocm on HPC
python training_engine/train_models.py
```

### Experiment Configuration (`experiments.json`, `train_models.py`)

All training runs are defined in `training_engine/experiments.json`. Each JSON object specifies one experiment. `train_models.py` merges each entry with `DEFAULT_CONFIG` and validates required fields before running.

**Required fields:**

| Field | Description |
|-------|-------------|
| `name` | Unique experiment identifier; used in output directory names |
| `model` | Architecture key — see Model Architectures below |
| `balancing` | Class imbalance strategy: `weighted_cost_function`, `oversampling`, or `none` |
| `data_path_key` | Dataset variant to use: `A`, `B`, `C`, `D`, or `E` |

**Key defaults (override in `experiments.json` as needed):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | `42` | Fixed seed for all splits, weight init, and shuffles |
| `EPOCHS` | `2` | Training epochs per fold |
| `BATCH_SIZE` | `4` | Mini-batch size for training |
| `N_SPLITS` | `5` | Number of folds for stratified k-fold cross-validation |
| `QUICK_TEST` | `True` | If `True`, runs only fold 0; useful for debugging |
| `HOLD_OUT_TEST_SET` | `True` | If `True`, reserves 20% of data as a final test set |
| `LEARNING_RATE` | `0.0001` | Adam optimiser learning rate |
| `DEVICE` | auto | `cuda` if a GPU is available, otherwise `cpu` |

### Step 1 — NIfTI File Discovery (`data_preprocess.py`)

`get_data_list` scans the dataset directory for `.nii` and `.nii.gz` files in the `0/` and `1/` subdirectories. Each file becomes a record with three fields: the absolute `image` path, the integer `label` (0 or 1), and the `path` filename (used for per-sample prediction logging).

### Step 2 — Hold-Out Test Set Split (`data_preprocess.py`)

Before any cross-validation, `split_data` reserves a stratified hold-out test set using `sklearn.model_selection.train_test_split`. By default, 20% of the full dataset is held out. Stratification ensures that the class ratio in the test set matches the full dataset.

**The test set is fixed for the entire experiment** — it is never seen during training or used to select hyperparameters. It is only evaluated once per fold using the best checkpoint at the very end.

### Step 3 — Stratified K-Fold Cross-Validation Split (`data_preprocess.py`)

The remaining 80% (the development set) is split by `StratifiedKFold` into `N_SPLITS` folds. For each fold iteration, one fold is the validation set and the rest form the training set. Stratification ensures each fold preserves the class distribution.

The full split hierarchy is:

```
All data (100%)
├── Test set (20%) — held out; evaluated once per fold with the best checkpoint
└── Development set (80%)
    ├── Fold 1 validation (16%)  ← train on remaining 4 folds, validate on this
    ├── Fold 2 validation (16%)
    ├── Fold 3 validation (16%)
    ├── Fold 4 validation (16%)
    └── Fold 5 validation (16%)
```

When `QUICK_TEST = True`, only fold 0 runs (used for fast debugging and CI).

### Step 4 — MONAI Transforms (`data_preprocess.py`)

MONAI's dictionary-based transform pipeline is applied lazily when each sample is loaded from disk by a DataLoader worker.

**Transforms applied to all splits (training, validation, test):**

| Transform | Purpose |
|-----------|---------|
| `LoadImaged` | Load the `.nii.gz` file from disk into a NumPy array |
| `EnsureChannelFirstd` | Add a channel dimension: `(D, H, W)` → `(1, D, H, W)` for single-channel input |
| `Resized` | Resize the volume to `(128, 128, 128)` to ensure a consistent input shape |
| `ScaleIntensityd` | Rescale voxel intensities to the range `[0, 1]` using per-volume min/max |
| `EnsureTyped` | Convert to a PyTorch tensor |

**Additional transforms applied to the training split only (augmentation):**

| Transform | Parameters | Purpose |
|-----------|------------|---------|
| `RandFlipd` | `prob=0.5`, axis 0 | Random horizontal flip — the brain is roughly symmetric, so flipped volumes are plausible |
| `RandRotate90d` | `prob=0.5`, up to 3 rotations | Random 90° rotation — invariance to scanner orientation |
| `RandGaussianNoised` | `prob=0.2`, `std=0.01` | Mild Gaussian noise — simulates scanner noise and prevents overfitting |

Augmentations are **not** applied to the validation or test sets, so evaluation metrics reflect performance on unaugmented data.

### Step 5 — Class Imbalance Handling (`data_preprocess.py`, `orchestrator.py`)

Aneurysm datasets are typically class-imbalanced (more healthy cases than aneurysm cases). Three strategies are supported, selected per-experiment via the `balancing` field:

**`weighted_cost_function`** — Computes inverse-frequency class weights from the training fold:

```
weight_class = total_samples / count_of_class
```

The weights are passed to `CrossEntropyLoss`, so the loss for a minority-class error is scaled up proportionally. Weights are computed from the **training fold only** to prevent label leakage.

**`oversampling`** — Uses PyTorch's `WeightedRandomSampler` to oversample minority-class examples during training. Each training sample is assigned a sampling weight inversely proportional to its class frequency, so on average each mini-batch sees a balanced class distribution. Sampling is done with replacement.

**`none`** — No rebalancing. Suitable for datasets with a moderate class ratio or when investigating baseline behaviour.

### Step 6 — Model Initialisation (`models.py`)

A fresh model is created at the start of **each fold** to prevent information leaking between folds through the model weights.

Available architectures (specified by the `model` field in `experiments.json`):

| Key | Architecture | Pretrained Weights |
|-----|-------------|-------------------|
| `R3D18` | torchvision R3D-18 (18-layer 3D ResNet) | Kinetics-400 video dataset; RGB first-layer weights averaged to 1 channel |
| `R3D50` | MONAI ResNet-50 (50-layer 3D ResNet, bottleneck) | MedicalNet (23 medical imaging datasets via Hugging Face Hub) |
| `DenseNet121` | MONAI DenseNet-121 | None — trained from scratch |
| `SwinUNETR` | MONAI SwinUNETR encoder + GAP + linear head | BTCV multi-organ segmentation (encoder only) |
| `UNet3D` | Custom 3D UNet (encoder + decoder + GAP head) | None — trained from scratch |
| `UNet3DWithBackbone` | 3D UNet with MONAI ResNet-18 encoder | MedicalNet encoder; decoder trained from scratch |
| `MIL_R3D18` | Attention MIL with MONAI ResNet-18 instance extractor | MedicalNet encoder |

All architectures accept a `(B, 1, 128, 128, 128)` input tensor (batch × channel × depth × height × width) and produce `(B, 2)` class logits.

For pretrained models with a 3-channel (RGB) first layer (torchvision R3D-18), single-channel adaptation is performed by **averaging the pretrained RGB channel weights**, which preserves the learned spatial filter structure instead of discarding it.

### Step 7 — Training Loop (`training.py`)

`train_one_epoch` runs one full pass over the training DataLoader:

1. Set the model to **training mode** (`model.train()`) — enables dropout and batch normalisation updates.
2. For each mini-batch:
   - Move inputs and labels to the target device (GPU or CPU).
   - Zero the accumulated gradients (`optimizer.zero_grad()`).
   - Run the forward pass to get class logits.
   - Compute the cross-entropy loss (optionally weighted for class imbalance).
   - Run the backward pass to compute gradients (`loss.backward()`).
   - Update model parameters (`optimizer.step()`).
3. Accumulate batch loss and correct prediction counts.
4. Return the epoch-level mean loss and accuracy.

The optimiser is **Adam** with a configurable learning rate (default `0.0001`).

### Step 8 — Validation Loop and F2-Score Checkpointing (`training.py`)

`validate_one_epoch` runs after each training epoch with `torch.no_grad()`:

1. Set the model to **evaluation mode** (`model.eval()`) — disables dropout and freezes batch norm statistics.
2. For each mini-batch, run the forward pass and collect predictions and ground-truth labels.
3. Compute epoch-level loss and accuracy.
4. Compute the **F2-score** across all batches:

```
F_β = (1 + β²) × (precision × recall) / (β² × precision + recall)    with β = 2
```

The F2-score weights recall twice as heavily as precision. In a medical screening context this is appropriate: a false negative (missing an aneurysm) is more clinically harmful than a false positive (unnecessary follow-up).

**Checkpointing:** `run_training_loop` keeps a deep copy of the model's `state_dict` at the epoch with the highest validation F2-score. This is the checkpoint that gets loaded for final evaluation, not the last epoch's weights.

### Step 9 — Final Evaluation (`orchestrator.py`)

After training completes for a fold, the best F2 checkpoint is restored and evaluated:

- If `HOLD_OUT_TEST_SET = True`, evaluation is run on the **held-out test set** (the 20% reserved before any splitting).
- If `HOLD_OUT_TEST_SET = False`, evaluation is run on the **validation fold**.

This evaluation produces per-sample predictions including filename, true label, and predicted label. These are used to generate all downstream artifacts.

### Step 10 — Artifact Generation (`plots.py`, `orchestrator.py`)

For each fold, the following artifacts are saved to `experiments/<name>/fold_<n>/`:

| File | Description |
|------|-------------|
| `best_model.pth` | PyTorch state dict of the F2-optimal checkpoint |
| `best_checkpoint_metadata.json` | Epoch number and F2-score of the saved checkpoint |
| `predictions.csv` | Per-sample filename, true label, and predicted label |
| `metrics.json` | Fold-level precision, recall, F1, F2, accuracy |
| `metrics_history.csv` | Per-epoch train/val loss, accuracy, and F2 |
| `confusion_matrix.png` | Raw count confusion matrix |
| `confusion_matrix_normalized.png` | Row-normalized confusion matrix (per-class recall rates) |
| `roc_curve.png` | ROC curve with AUC computed from softmax probabilities |

After all folds complete, `evaluate_models` aggregates predictions across folds and writes `experiments/<name>/<name>_evaluation_summary.csv` — a cross-fold summary of precision, recall, F1, F2, and accuracy.

---

## Running the Full Pipeline

### 1. Prepare data (local only)

```bash
cd data_engine
source .venv/bin/activate

python data_cleaner.py \
    --raw-dir /mnt/data/cases-3/raw \
    --nifti-dir /mnt/data/cases-3/nifti

python dataset_gen.py --workers 3
```

### 2. Transfer datasets to HPC (if using the cluster)

```bash
rsync -avz /mnt/data/cases-3/ user@hpc-cluster:/path/to/data/
```

Then update `data_path_key` paths in `experiments.json` to point to the cluster location.

### 3. Configure experiments

Edit `training_engine/experiments.json`. Minimal example:

```json
[
  {
    "name": "r3d18_weighted_datasetA",
    "model": "R3D18",
    "balancing": "weighted_cost_function",
    "data_path_key": "A",
    "EPOCHS": 30,
    "QUICK_TEST": false
  }
]
```

### 4. Run training

```bash
cd training_engine
source .venv/bin/activate

# Local
python train_models.py

# HPC — ROCm cluster (from repository root)
sbatch infrastructure/training/start_script_rocm.slurm
```

Results are saved to `experiments/<name>/`.

### 5. Running tests and linting

```bash
# Data engine
cd data_engine && pytest
cd data_engine && ruff check .

# Training engine (no GPU or dataset required)
cd training_engine && pytest
cd training_engine && ruff check .
```

---

## Project Structure

```
aneurysm_cnn/
├── data_engine/                    # Stage 1: DICOM → NIfTI pipeline
│   ├── data_cleaner.py             # Pipeline entry point (steps 1–5)
│   ├── dataset_gen.py              # Dataset variant generator (step 6)
│   ├── diagnostics/                # One-time and ad-hoc analysis scripts
│   │   └── check_nifti_quality.py  # Ad-hoc quality inspection script
│   ├── dataset/
│   │   └── classes.csv             # Ground-truth aneurysm labels (tracked in git)
│   ├── output/                     # GITIGNORED: pipeline runtime artifacts
│   ├── src/
│   │   ├── file_utils.py           # Folder discovery and name standardization
│   │   ├── dicom_utils.py          # DICOM validation and mixed-series analysis
│   │   ├── class_utils.py          # Class label joining and missing-class detection
│   │   ├── nifti_utils.py          # DICOM → NIfTI conversion
│   │   ├── nifti_resize.py         # Five dataset variant functions (A–E)
│   │   └── logging_utils.py        # Logger setup and audit log writing
│   └── tests/
│
├── training_engine/                # Stage 2: model training pipeline
│   ├── train_models.py             # Entry point: loads experiments.json, runs all experiments
│   ├── experiments.json            # Experiment registry
│   └── src/
│       ├── data_preprocess.py      # File discovery, splitting, transforms, DataLoaders
│       ├── models.py               # Model factory: R3D18, R3D50, DenseNet, SwinUNETR, UNet3D, MIL
│       ├── training.py             # Per-epoch train/validate loops; F2 checkpointing
│       ├── orchestrator.py         # k-fold experiment orchestration; fold result aggregation
│       └── plots.py                # Confusion matrices, ROC curves, CSVs, metric summaries
│
├── infrastructure/                 # Environment configs and HPC scripts
│   ├── data/
│   │   ├── environment_data.yml    # Local, data engine dependencies
│   │   ├── environment_data_hpc.yml# HPC, data engine (Conda)
│   │   ├── dataset_gen*.slurm      # HPC scripts for data generation
│   │   ├── transfer_datasets.sh
│   │   └── transfer_nifti.sh
│   └── training/
│       ├── environment_local.yml   # Local, training engine dependencies
│       ├── environment_hpc.yaml    # HPC, training engine (Conda)
│       ├── start_script_rocm.slurm # HPC, ROCm cluster
│       └── test_rocm.py            # HPC diagnostics
│
├── experiments/                    # GITIGNORED: all training run outputs
├── logs/                           # GITIGNORED: cross-cutting runtime logs
├── specs/                          # Feature specifications
├── CLAUDE.md
└── README.md
```
