# Pipeline Overview

This project is a two-stage pipeline for binary classification of intracranial aneurysms from 3D medical imaging data. Raw DICOM scans are first standardised into NIfTI volumes, then used to train and evaluate 3D convolutional neural networks.

```
Raw DICOM files
      │
      ▼
┌─────────────────┐
│   Data Engine   │  Validates, converts, and preprocesses imaging data
└─────────────────┘
      │
      ▼
NIfTI datasets (4 preprocessing variants)
      │
      ▼
┌─────────────────────┐
│   Training Engine   │  Trains and evaluates 3D CNN models
└─────────────────────┘
      │
      ▼
Experiment results (weights, metrics, plots)
```

---

## Stage 1 — Data Engine

Entry point: [`data_engine/data_cleaner.py`](data_engine/data_cleaner.py)

Orchestrates a sequential pipeline that takes a directory of raw DICOM cases and produces a set of labelled NIfTI volumes ready for training.

### Step 1 — Folder discovery and standardisation
**[`data_engine/src/file_utils.py`](data_engine/src/file_utils.py)**

Scans the raw data directory, standardises case folder names to a canonical format (e.g. `BP001`), and classifies each folder by its structure (flat DICOM files, a single subfolder, or multiple subfolders). Produces a per-case data dictionary that all subsequent steps operate on.

### Step 2 — DICOM validation
**[`data_engine/src/dicom_utils.py`](data_engine/src/dicom_utils.py)**

Validates each DICOM series through 11 checks: orientation consistency, slice spacing regularity, duplicate detection, scout image removal, minimum slice count, and more. Slices are sorted by physical position (projection onto the normal vector) rather than `InstanceNumber`, which is unreliable across scanners. Cases that fail any check are marked with a specific status code and excluded from conversion.

### Step 3 — Label assignment
**[`data_engine/src/class_utils.py`](data_engine/src/class_utils.py)**

Joins each case with its ground-truth aneurysm label (0 = negative, 1 = positive) from `dataset/classes.csv`. Cases with no matching label are flagged and excluded from conversion.

### Step 4 — NIfTI conversion
**[`data_engine/src/nifti_utils.py`](data_engine/src/nifti_utils.py)**

Converts validated DICOM series to NIfTI (`.nii.gz`) using ITK and MONAI. The affine matrix encoding voxel-to-world-space coordinates is preserved. Conversion runs in parallel across worker processes (not threads, to bypass the GIL for CPU-bound ITK work). Output is written to `nifti/0/` and `nifti/1/` subdirectories reflecting the class label.

### Step 5 — Audit logging
**[`data_engine/src/logging_utils.py`](data_engine/src/logging_utils.py)**

Writes structured JSONL logs (one entry per event, DEBUG level) and a human-readable audit CSV summarising the conversion outcome for every case. The JSONL format is chosen over plain text so logs can be queried programmatically.

---

Entry point: [`data_engine/dataset_gen.py`](data_engine/dataset_gen.py)

Reads the NIfTI output from `data_cleaner.py` and generates four preprocessing variants in parallel, each encoding a different trade-off between spatial resolution, field of view, and number of interpolation passes.

### Step 6 — Dataset variant generation
**[`data_engine/src/nifti_resize.py`](data_engine/src/nifti_resize.py)**

Each NIfTI file is loaded once and passed through all four variant transforms, avoiding redundant disk reads. The four variants form a 2×2 grid across two independent decisions:

| | No resampling (native spacing) | Isotropic resampling |
|---|---|---|
| **Crop / pad to 128³** | **C** — native spacing, centre-crop or zero-pad | **A** — resample to 1 mm isotropic, then crop/pad |
| **Shrink to 128³** | **D** — native spacing, trilinear resize | **B** — isotropic resample (largest dim → 128 voxels), then zero-pad |

> Variant B is generally most principled: one interpolation pass to isotropic spacing, aspect ratio preserved, no anatomy discarded by cropping.

---

## Stage 2 — Training Engine

Entry point: [`training_engine/train_models.py`](training_engine/train_models.py)

Reads experiment definitions from [`experiments.json`](training_engine/experiments.json), merges each with a set of defaults, and runs them sequentially through the orchestrator.

### Step 7 — Data loading and splitting
**[`training_engine/src/data_preprocess.py`](training_engine/src/data_preprocess.py)**

Discovers NIfTI files from the selected dataset variant and performs a two-level split: 20% is held out as a test set before any model sees the data; the remaining 80% is used for stratified k-fold cross-validation. MONAI dictionary-based transforms handle loading, channel insertion, intensity normalisation, and (for training folds only) stochastic augmentation. Class imbalance is addressed either via `WeightedRandomSampler` (oversampling) or class-weighted loss, depending on the experiment config.

### Step 8 — Model construction
**[`training_engine/src/models.py`](training_engine/src/models.py)**

Factory function `get_model` constructs one of nine supported architectures:

| Key | Architecture | Pretrained on |
|---|---|---|
| `R3D18` | torchvision R3D-18 | Kinetics-400 (video) |
| `R3D50` | MONAI ResNet-50 | MedicalNet (medical imaging) |
| `R3D10` | MONAI ResNet-10 | MedicalNet |
| `DenseNet121` | MONAI DenseNet-121 | from scratch |
| `SwinUNETR` | MONAI SwinUNETR | from scratch |
| `UNet3D` | Custom UNet3D | from scratch |
| `UNet3DWithBackbone` | UNet3D + ResNet-18 backbone | MedicalNet |
| `MIL_R3D18` | Multiple Instance Learning wrapper | MedicalNet |
| `UNETR` | MONAI UNETR (ViT encoder) | from scratch |

All models are adapted for single-channel (greyscale) input. When `USE_TABULAR=True`, a `MultiModalWrapper` fuses image features with patient metadata (age, gender) via late fusion before the classification head.

### Step 9 — Training loop
**[`training_engine/src/training.py`](training_engine/src/training.py)**

Runs per-epoch train and validation passes. The best model checkpoint is saved based on **F2-score** (not accuracy or F1) — F2 weights recall twice as heavily as precision, reflecting the clinical cost asymmetry where a missed aneurysm (false negative) is more dangerous than a false alarm (false positive). Early stopping halts training if F2 does not improve for a configurable number of epochs.

### Step 10 — Fold orchestration
**[`training_engine/src/orchestrator.py`](training_engine/src/orchestrator.py)**

Manages the outer k-fold loop. For each fold: initialises a fresh model (same random seed, independent weights), builds dataloaders, runs the training loop, restores the best checkpoint, evaluates on the test or validation set, saves weights and per-fold metrics, then frees GPU memory before the next fold. Class weights are computed inside the fold loop from training data only, preventing label leakage from the validation fold.

### Step 11 — Evaluation and plotting
**[`training_engine/src/plots.py`](training_engine/src/plots.py)**

Aggregates results across folds and produces:
- Confusion matrices (row-normalised, per fold and aggregate)
- ROC curves with AUC (requires re-running inference to obtain class probabilities)
- Per-fold and mean ± std metric tables (F2, F1, precision, recall, accuracy, AUC)
- CSV files for training history and final evaluation metrics
- Prediction audit CSVs (per-sample true label, predicted label, confidence)

---

## Optional Feature — Tabular Metadata Fusion

The pipeline supports fusing patient metadata (age, gender) with image features using a **late-fusion** architecture. This is an opt-in feature controlled by two fields in `experiments.json`:

```json
"USE_TABULAR": true,
"TABULAR_CSV": "/path/to/metadata.csv"
```

The CSV must have columns: `case_id`, `age`, `gender` — where `case_id` matches the NIfTI filename without its extension (e.g. `BP001` for `BP001.nii.gz`). Age is normalised to `[0, 1]` by dividing by 100; gender is encoded as `0.0` / `1.0`.

### How it flows end-to-end

```
experiments.json (USE_TABULAR=true, TABULAR_CSV=...)
        │
        ▼
train_models.py         — validates that TABULAR_CSV is set when USE_TABULAR=true
        │
        ▼
orchestrator.run_experiment
  ├── get_data_list(..., tabular_csv=...)    — attaches a float32 [age/100, gender]
  │                                            array to each record dict under "tabular"
  ├── build_dataloaders(..., use_tabular=True) — includes "tabular" key in each batch
  └── get_model(..., use_tabular=True)       — wraps the CNN in MultiModalWrapper
        │
        ▼
training.train_one_epoch / validate_one_epoch
  — detects "tabular" key in batch at runtime
  — calls model(image, tabular) instead of model(image)
```

### Architecture — `MultiModalWrapper`
**[`training_engine/src/models.py`](training_engine/src/models.py)**

```
image ──► CNN backbone (classifier stripped) ──► image features (e.g. 512-d)
                                                          │
                                                          ▼
                                                    concat ──► fusion MLP ──► logits
                                                          ▲
tabular ──► small MLP (2 → 16-d) ──────────────────────►─┘
```

The tabular branch is intentionally shallow (two linear layers, 16-d output) to avoid overfitting on only two input features. The fusion head is a dropout MLP that maps the concatenated vector to class logits. The `_strip_classifier` helper removes the original classification head from the CNN before wrapping, so the backbone outputs a feature vector rather than logits.
