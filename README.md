# CNNs for Intracranial Aneurysm Detection on TOF-MRA

A master's thesis project implementing a full end-to-end pipeline for training and evaluating 2D and 3D CNNs that detect intracranial aneurysms in TOF-MRA (time-of-flight magnetic resonance angiography) exams, with exam-level labels only.

The system is split into two independent stages: a **data engine** that converts raw DICOM archives into standardised, resampled NIfTI volumes and 2D representations (MIP, thin-slab, Frangi), and a **training engine** that trains, cross-validates, and evaluates classification models across configurable experiments. Both engines are thin project-specific wrappers around two reusable libraries:

| Library | Role | Source |
|---------|------|--------|
| [**medprep**](https://github.com/bruno-schenberg/medprep) | Medical image preprocessing (DICOM I/O, NIfTI conversion, resampling, MIP generation, QC) | Extracted from `data_engine/` |
| [**dlorchestra**](https://github.com/bruno-schenberg/dlorchestra) | DL experiment orchestration (k-fold CV, training loop, metrics, artifacts, model registry) | Extracted from `training_engine/` |

---

## Results

Private single-hospital dataset: 759 TOF-MRA exams (155 with aneurysm). 639 exams for 5-fold cross-validation (dev) and a frozen 114-exam hold-out test set (23 positives). Every choice was made on validation; the test set was only reported. AUC is the mean over the 5 fold models unless noted; brackets are 95 % bootstrap CIs.

| Model | Config | Val AUC | Test AUC |
|-------|--------|---------|----------|
| **2D**: ResNet-50 late fusion on 512² axial/coronal/sagittal MIPs of a 90 mm vessel-anchored window | `configs/final/2D-3_512.json` | 0.762 | 0.621 |
| **3D**: R3D-18 (Kinetics) on 256³ isotropic volumes, + age/sex | `configs/final/3D-1_r3d18k_iso256_mm.json` | 0.742 | 0.605 |
| nnU-Net zero-shot (published multi-centre model, never trained on this data) | `segmentation/` | 0.740 (dev) | 0.680 |
| **Ensemble** 2D + 3D + nnU-Net (stacked percentile ranks, scored once on test) | `analysis/ensemble.py` | 0.847 (dev OOF) | **0.723** [0.597, 0.837] |

- 3D did not outperform 2D: 2D is numerically higher, but no paired difference is significant.
- In 2D, the input window matters most, then late fusion and 512² resolution. N4 bias correction, age/sex, extra public data (Lausanne, OpenNeuro ds003949), attention models, 768² inputs and other backbones did not help.
- No whole-volume 3D backbone reliably detected an aneurysm-sized synthetic sphere, which points to global pooling as the limit for 3–5 mm lesions.
- Every model loses about 0.12–0.15 AUC from validation to test, and with 23 test positives the CIs are about ±0.11 wide, so most comparisons are underpowered.

---

## Pipeline Overview

```
Raw DICOM archives (.zip)
        |
        v
+-------------------------------+
|      Stage 1: Data Engine     |
|                               |
|  0. Archive extraction        |
|  1. Folder discovery & naming |
|  2. Metadata join             |
|  3. DICOM QC + series select  |
|  4. NIfTI conversion          |
|  5. Centroid crop (180 mm)    |
|  6. Dataset variant gen       |
|  7. 2D representations (MIP)  |
+---------------+---------------+
                |  /mnt/data/datasets/
                |  iso_128/  iso_256/  mip_*/
                v
+-------------------------------+
|    Stage 2: Training Engine   |
|                               |
|  1. File discovery            |
|  2. Hold-out test split       |
|  3. Stratified k-fold split   |
|  4. Transforms (MONAI/torchv) |
|  5. DataLoader assembly       |
|  6. Model initialisation      |
|  7. Training loop (AMP)       |
|  8. Validation + F2 ckpt      |
|  9. Final evaluation          |
| 10. Artifact generation       |
+-------------------------------+
```

---

## Core Dependencies

| Library | Role |
|---------|------|
| **medprep** | DICOM I/O, NIfTI conversion, resampling, QC, MIP generation |
| **dlorchestra** | Experiment orchestration, training loop, metrics, registry |
| **PyTorch** | Deep learning framework; training loop, optimiser, loss |
| **MONAI** | Medical imaging transforms, 3D model architectures |
| **torchvision** | 2D models (ResNet-50, EfficientNet, ConvNeXt, ViT) and R3D-18 video model |
| **NiBabel** | Reading and writing NIfTI (`.nii.gz`) volumetric files |
| **pydicom** | Parsing and validating raw DICOM files |
| **scikit-learn** | Stratified splitting, F-beta score, ROC curve metrics |
| **SciPy** | Trilinear resampling (`ndimage.zoom`) for dataset generation |

---

## Environment Setup

Requirements: Python ≥ 3.10 and, for training, a CUDA GPU (the 2D models were trained on a 12 GB card with AMP and gradient accumulation; the 3D models at 256³ need far more memory). Install the PyTorch build matching your CUDA version first if the default wheel doesn't fit your system.

Each engine is a Python package with its own `pyproject.toml` and virtual environment. Libraries are installed as dependencies automatically.

```bash
# Data Engine
cd data_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
deactivate
cd ..

# Training Engine
cd training_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
deactivate
cd ..
```

Tests (CPU, synthetic data, no dataset needed):

```bash
(cd training_engine && pip install -e ".[dev]" && python -m pytest tests)
(cd data_engine && pip install -e ".[dev]" && python -m pytest)
```

---

## Data

**No data is included** — not the images, labels, splits or trained weights. The code is published so the pipeline can be read and reused on your own data.

What the code expects:

| Input | Default location | Override |
|-------|------------------|----------|
| Raw DICOM archives (`.zip`, one folder per exam) | `/mnt/data/zips` | `DATA_ROOT` or `SOURCE_DIR` (see [`data_engine/config.py`](data_engine/config.py)) |
| Exam table: `exam,class,location,Age,gender,brand,scanner,field_strength_T` (one row per exam; `class` 0 = healthy, 1 = aneurysm; other values are not trained on) | `dataset/dataset.csv` | `DATASET_CSV` |
| Generated datasets (one directory per variant) | `/mnt/data/datasets/<variant>` | `DATASETS_ROOT` or per-variant `DATASET_*` (see [`training_engine/config.py`](training_engine/config.py)) |

Optional, set per experiment config:

- `TEST_IDS_FILE`: JSON list of exam IDs kept as a frozen hold-out test set.
- `EXCLUSIONS_FILE`: JSON with `exclude` (exam → reason) and `patient_groups` (lists of exams from the same patient, kept in one fold).

`dataset/` is gitignored so your own data stays out of version control.

---

## Stage 1: Data Engine

The data engine lives in `data_engine/` and runs on the local development machine. It turns a directory of raw DICOM archives into clean, resampled NIfTI volumes and 2D representations, organised by class label.

Entry point: `data_engine/scripts/run_pipeline.py`

```bash
cd data_engine
source .venv/bin/activate

# Full pipeline (all steps)
python scripts/run_pipeline.py

# Single step
python scripts/run_pipeline.py --step 1

# Multiple steps
python scripts/run_pipeline.py --step 0 1 2

# Dry run (skip extraction + conversion)
python scripts/run_pipeline.py --step 1 --dry-run

# Dataset generation — specific variants only
python scripts/run_pipeline.py --step 2 --variants iso_128

# Test mode — uses sample data under /mnt/data/sample_*
python scripts/run_pipeline.py --test --step 0 1 2 --variants iso_128
```

| Flag | Description |
|------|-------------|
| `--step N [N ...]` | Run specific steps (default: all 0-3) |
| `--dry-run` | Skip extraction (step 0) and conversion (step 1) |
| `--workers N` | Parallel workers for conversion/resampling (default: 4) |
| `--variants NAME [...]` | Step 2: variant names to generate (default: all) |
| `--sample` | Step 2: restrict to exams in `dataset/sample_exams.csv` (you provide it) |
| `--test` | Redirect all paths to sample dirs under `/mnt/data/` |

Detailed documentation: [`docs/data_engine.md`](docs/data_engine.md)

---

## Stage 2: Training Engine

The training engine lives in `training_engine/` and handles experiment configuration, data loading, model training, cross-validation, and artifact generation.

Entry point: `training_engine/train_models.py`

```bash
cd training_engine
source .venv/bin/activate

# Run experiments
python train_models.py --experiments configs/smoke_test.json
```

| Flag | Description |
|------|-------------|
| `--experiments PATH` | Path to experiment JSON config file (required) |

### Experiment Configuration

All training runs are defined as flat JSON lists. Each entry is merged over `DEFAULT_CONFIG` in [`config.py`](training_engine/config.py). Minimal example:

```json
[
  {
    "name": "r3d18_weighted_iso128",
    "model": "R3D18",
    "data_path_key": "iso_128",
    "EPOCHS": 100,
    "EARLY_STOPPING_PATIENCE": 15
  }
]
```

**Required fields:**

| Field | Description |
|-------|-------------|
| `name` | Unique experiment identifier; used in output directory names |
| `model` | Architecture key (see below) |
| `data_path_key` | Dataset variant: `iso_128`, `iso_256`, `sample_iso_128`, or MIP variants |

`configs/` holds the experiment files used during the thesis; `configs/final/` is the final campaign (generated by `scripts/make_final_configs.py`). The final 2D model is `configs/final/2D-3_512.json`.

### Model Architectures

**3D models (volumetric input):**

| Key | Architecture | Pretrained on |
|-----|-------------|---------------|
| `R3D10` | MONAI ResNet-10 | MedicalNet |
| `R3D18` | MONAI ResNet-18 | MedicalNet |
| `R3D18_KINETICS` | torchvision R3D-18 | Kinetics-400 (video) |
| `R3D50` | MONAI ResNet-50 | MedicalNet |
| `DENSENET121` | MONAI DenseNet-121 | from scratch |
| `NNUNET_ENC` | nnU-Net encoder + GAP | nnU-Net segmentation (Dataset610) |
| `NNUNET_ENC_MULTISCALE` | nnU-Net encoder, stages 2-5 | nnU-Net segmentation |
| `NNUNET_ENC_ATTN` | nnU-Net encoder + spatial attention | nnU-Net segmentation |
| `NNUNET_ENC_MULTISCALE_ATTN` | Multi-scale + attention | nnU-Net segmentation |
| `MIL_R3D18K` | Gated attention MIL (3D patches) | Kinetics-400 |
| `SE_R3D18` | R3D18-Kinetics + squeeze-excitation channel attention | Kinetics-400 |
| `SWIN_ENC` | MONAI SwinUNETR encoder + GAP head | from scratch, or MONAI self-supervised SwinViT checkpoint via `SWIN_CHECKPOINT` |
| `VIT3D_B32` | MONAI ViT-B/32 (3D), 32<sup>3</sup> patches | from scratch |
| `PATCH_CLASSIFIER_3D` | Small 4-block 3D CNN for nnU-Net candidate patches (Phase 7A cascade stage 2, expects 64<sup>3</sup> patches, not full volumes) | from scratch |

**2D models (MIP classification):**

| Key | Architecture | Pretrained on |
|-----|-------------|---------------|
| `RESNET50_2D` | ResNet-50 (1ch) | ImageNet |
| `RESNET50_2D_2CH` | ResNet-50 (2ch, Frangi) | ImageNet |
| `RESNET50_2D_3CH` | ResNet-50 (3ch, early fusion) | ImageNet |
| `RESNET50_2D_NCH` | ResNet-50 (N-ch, thin-slab/multi-angle) | ImageNet |
| `RESNET50_2D_LATEFUSION` | 3x ResNet-50 (late fusion) | ImageNet |
| `EFFICIENTNET_B4_3CH` | EfficientNet-B4 (3ch) | ImageNet |
| `CONVNEXT_T_3CH` | ConvNeXt-Tiny (3ch) | ImageNet |
| `VIT_B16_3CH` | ViT-B/16 (3ch) | ImageNet |

**Slice-level aggregation (3D volume as 2D slice sequence):**

| Key | Architecture | Pretrained on |
|-----|-------------|---------------|
| `CONVLSTM` | ResNet-18 + BiLSTM | ImageNet |
| `MIL_SLICE_2D` | ResNet-18 + gated attention | ImageNet |

### Multimodal Fusion

When `USE_MULTIMODAL=true`, a `MultiModalWrapper` fuses image features with patient metadata (age, gender) via late fusion before the classification head. Metadata is loaded from `dataset/dataset.csv`.

---

## Segmentation

The `segmentation/` directory evaluates a pretrained segmentation model as a zero-shot baseline against the classifiers: the multi-centric nnU-Net from Zenodo (record 13386859, Dataset610).

- `setup.sh` installs nnU-Net and downloads the weights; `run_inference.sh` runs inference and evaluation.
- `evaluate.py` turns segmentations into exam-level scores; `extract_candidates.py`, `build_patch_dataset.py` and `feature_classifier.py` build a candidate-patch cascade on top of them.

---

## Running the Full Pipeline

### 1. Prepare data (local)

```bash
cd data_engine && source .venv/bin/activate
python scripts/run_pipeline.py --step 0 1 2 3
```

### 2. Configure and run experiments

```bash
cd training_engine && source .venv/bin/activate
python train_models.py --experiments configs/my_experiment.json
```

Results are saved to `training_engine/experiments/<name>/`.

### Quick smoke test (sample data)

```bash
# 1. Build sample zips from the full dataset (one-time)
cd data_engine && source .venv/bin/activate
python scripts/make_sample.py

# 2. Run data pipeline on sample data
python scripts/run_pipeline.py --test --step 0 1 2 --variants iso_128

# 3. Smoke-test training
cd ../training_engine && source .venv/bin/activate
python train_models.py --experiments configs/smoke_test.json
```

---

## Project Structure

```
cnn_for_aneurysm/
├── data_engine/                       # Stage 1: DICOM -> NIfTI pipeline
│   ├── pyproject.toml                 # Package definition (depends on medprep)
│   ├── config.py                      # Paths, QC thresholds, variant definitions
│   ├── scripts/
│   │   ├── run_pipeline.py            # Pipeline orchestrator (steps 0-3)
│   │   ├── make_sample.py             # Build sample zips for testing
│   │   └── justification/             # Design decision justification scripts
│   ├── steps/
│   │   ├── step0/                     # Extract, discover, name (project-specific)
│   │   ├── step1/                     # Metadata, QC, convert (calls medprep)
│   │   ├── step2/                     # Resample to dataset variants (calls medprep)
│   │   ├── step3/                     # Visual QC
│   │   └── step_mip/                  # 2D representation generation (MIP, thin-slab, Frangi)
│   └── reports/                       # Pipeline output CSVs (gitignored)
│
├── training_engine/                   # Stage 2: model training pipeline
│   ├── pyproject.toml                 # Package definition (depends on dlorchestra)
│   ├── config.py                      # Paths, DEFAULT_CONFIG, device, validate_config
│   ├── train_models.py                # CLI entry point
│   ├── configs/                       # Experiment JSON files
│   ├── src/
│   │   ├── data.py                    # File discovery, transforms, DataLoaders
│   │   ├── models.py                  # Model architectures (registry-based)
│   │   ├── split.py                   # Grouped, stratified splits; frozen test set; exclusions
│   │   └── experiment.py              # AneurysmExperiment (thin wrapper over dlorchestra)
│   ├── analysis/                      # Ensembling, statistics, figures, cross-model comparison
│   ├── scripts/                       # Config generation, dataset checks, diagnostics
│   └── tests/                         # Model, split, cache, overfit and run-contract tests
│
├── segmentation/                      # nnU-Net zero-shot baseline and candidate cascade
│
├── dataset/                           # Your exam table and split files (gitignored, not included)
│
├── docs/                              # Design documentation
│   ├── data_engine.md                 # Data engine design document
│   ├── augmentation.md                # Training augmentation rationale
│   ├── justification_scripts.md       # Scripts behind the design decisions
│   └── aneurysm_size_reference.md     # Voxel size vs. aneurysm size per variant
│
├── PIPELINE.md                        # Technical pipeline reference
└── README.md
```

---

## License

MIT, see [`LICENSE`](LICENSE). `medprep` and `dlorchestra` are MIT-licensed as well.
