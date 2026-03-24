# 3D CNN for Aneurysm Detection

This project provides a complete pipeline for training and evaluating 3D Convolutional Neural Networks (CNNs) for aneurysm detection in medical imaging data. The workflow is designed to be modular and experiment-driven, allowing for easy configuration and comparison of different models, data preprocessing techniques, and balancing strategies.

## Features

*   **Modular Data Pipeline**: Separate scripts for cleaning raw data and generating training-ready datasets.
*   **Experiment-Driven Training**: Define and manage multiple training runs through a simple `experiments.json` configuration file.
*   **Configurable Training**: Easily adjust hyperparameters, model architecture, data balancing, and dataset choice for each experiment.
*   **Cross-Validation**: Built-in support for k-fold cross-validation to ensure robust model evaluation.
*   **Hold-Out Test Set**: Option to reserve a final test set for unbiased performance assessment after model selection.
*   **Reproducibility**: Track experiments with a fixed random seed and organized output directories.

## Core Libraries

*   **PyTorch**: The primary deep learning framework used for building and training the 3D CNN models.
*   **MONAI**: A PyTorch-based framework for deep learning in healthcare imaging, used for data loading, transformations, and models.
*   **NiBabel**: For reading and writing medical imaging formats, specifically NIfTI (`.nii.gz`).
*   **pydicom**: Used in the `data_cleaner.py` script to parse and validate raw DICOM files.
*   **NumPy**: For numerical operations, particularly in image manipulation and data preprocessing.

## Environment Setup

Three focused conda environments cover all use cases. No `requirements.txt` files are used — all dependencies are managed by conda.

| Environment | YAML | Where used | Pipeline stage |
|-------------|------|------------|----------------|
| `aneurysm_cnn_data` | `environment/data/environment_data.yml` | Local | Data engine only |
| `aneurysm_cnn_cuda` | `environment/cuda/environment_cuda.yml` | Local | Training engine (CUDA) |
| `aneurysm_cnn_rocm` | `environment/rocm/environment_rocm.yaml` | HPC cluster | Training engine (ROCm) |

```bash
# Local — data engine
conda env create -f environment/data/environment_data.yml
conda activate aneurysm_cnn_data

# Local — training engine (CUDA)
conda env create -f environment/cuda/environment_cuda.yml
conda activate aneurysm_cnn_cuda

# HPC cluster — training engine (ROCm)
conda env create -f environment/rocm/environment_rocm.yaml
conda activate aneurysm_cnn_rocm
```

## Project Structure

```
aneurysm_cnn/
├── data_engine/                    # Stage 1: DICOM → NIfTI pipeline
│   ├── data_cleaner.py             # Pipeline entry point
│   ├── dataset_gen.py              # Dataset variant generator
│   ├── check_nifti_quality.py      # Ad-hoc quality inspection script
│   ├── pytest.ini                  # Test runner configuration
│   ├── dataset/                    # TRACKED: static source assets
│   │   └── classes.csv             # Ground-truth aneurysm labels
│   ├── output/                     # GITIGNORED: generated pipeline artifacts
│   ├── src/                        # Library modules
│   └── tests/                      # Test suite
│
├── training_engine/                # Stage 2: model training pipeline
│   ├── train_models.py             # Training entry point
│   ├── experiments.json            # Experiment configuration registry
│   ├── datasets/                   # GITIGNORED: preprocessed NIfTI variants
│   └── src/                        # Library modules
│
├── environment/                    # Conda environment definitions ONLY
│   ├── data/
│   │   └── environment_data.yml    # Local, data engine (Python 3.11)
│   ├── cuda/
│   │   └── environment_cuda.yml    # Local, training engine (Python 3.11, CUDA 12.1)
│   └── rocm/
│       └── environment_rocm.yaml   # HPC, training engine (Python 3.10, ROCm 6.0)
│
├── hpc/                            # HPC job submission scripts ONLY
│   ├── cuda/
│   │   └── start_script_cuda.slurm
│   └── rocm/
│       ├── start_script_rocm.slurm
│       ├── test_rocm.slurm
│       └── test_rocm.py
│
├── experiments/                    # GITIGNORED: all training run outputs
├── logs/                           # GITIGNORED: cross-cutting runtime logs
├── specs/                          # Feature specifications
├── .gitignore
├── CLAUDE.md
└── README.md
```

## Workflow

### 1. Data Preparation (local only)

The data engine runs exclusively on the local development machine. NIfTI output is written to the external SSD at `/mnt/data/cases-3/`.

```bash
conda activate aneurysm_cnn_data

# Full DICOM → NIfTI pipeline
python data_engine/data_cleaner.py --raw-dir /mnt/data/cases-3/raw --nifti-dir /mnt/data/cases-3/nifti

# Generate the 5 dataset preprocessing variants
python data_engine/dataset_gen.py

# Run tests
cd data_engine && pytest

# Lint
cd data_engine && ruff check .
```

Dataset variants are written to `/mnt/data/cases-3/` on the external SSD — never inside the repository.

### 2. Transfer Datasets to HPC (if using cluster)

After running the data engine locally, sync the processed datasets to the HPC cluster before submitting training jobs:

```bash
rsync -avz /mnt/data/cases-3/datasets/ user@hpc-cluster:/path/to/datasets/
```

Then update `training_engine/experiments.json` on the cluster to point `data_path` at the transferred dataset location.

### 3. Configure Experiments

All training runs are defined in `training_engine/experiments.json`. Each object in the JSON array represents a single experiment.

**Required fields:**
- `name` — unique experiment identifier
- `model` — architecture: `R3D18`, `R3D50`
- `balancing` — strategy: `weighted_cost_function`, `oversampling`, `none`
- `data_path_key` — dataset variant: `A`, `B`, `C`, `D`, or `E`

**Optional overrides:** `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`, `N_SPLITS`, `QUICK_TEST`, `HOLD_OUT_TEST_SET`

### 4. Run Training

```bash
conda activate aneurysm_cnn_cuda   # or aneurysm_cnn_rocm on HPC

# Run all experiments (from repository root)
python training_engine/train_models.py
```

Results are saved to `experiments/<name>/` at the repository root.

### 5. HPC Job Submission

All SLURM scripts are in `hpc/`:

```bash
# ROCm cluster
sbatch hpc/rocm/test_rocm.slurm
sbatch hpc/rocm/start_script_rocm.slurm

# CUDA cluster
sbatch hpc/cuda/start_script_cuda.slurm
```

## Configuration Details

Key default parameters in `train_models.py`:

*   `RANDOM_SEED`: 42
*   `QUICK_TEST`: `True` (single fold, fast debug)
*   `HOLD_OUT_TEST_SET`: `True` (20% reserved for final evaluation)
*   `EPOCHS`: 2, `BATCH_SIZE`: 4, `N_SPLITS`: 2
