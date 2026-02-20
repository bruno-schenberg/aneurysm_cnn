# Implementation Plan: Aneurysm Experiment Pipeline

**Branch**: `001-aneurysm-experiment-pipeline` | **Date**: 2026-02-18 | **Spec**: [specs/001-aneurysm-experiment-pipeline/spec.md](spec.md)
**Input**: Feature specification from `/specs/001-aneurysm-experiment-pipeline/spec.md`

## Summary

Implement a scientifically rigorous, modular deep learning pipeline for aneurysm detection in 3D medical imaging. The system consists of a **Data Engine** for deterministic preprocessing (DICOM → NIfTI with 5 variant strategies) and a **Training Engine** for config-driven, reproducible model training (PyTorch/MONAI) with k-fold cross-validation and automated artifact generation.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**:
*   **Deep Learning**: PyTorch 2.x, MONAI 1.3+ (Core framework)
*   **Imaging I/O**: NiBabel (NIfTI handling), Pydicom (DICOM parsing)
*   **Data/Math**: NumPy, Pandas, Scikit-learn (Metrics, K-Fold)
*   **Visualization**: Matplotlib
**Storage**: Local filesystem (SSD/NVMe recommended for training data)
**Testing**: Pytest (Unit tests), Integration scripts
**Target Platform**: Linux (Local Dev + HPC Clusters via SLURM)
**Hardware Acceleration**: NVIDIA CUDA (12.1+) or AMD ROCm (6.0+)
**Project Type**: Python CLI Application (Dual Engine: Data & Training)
**Performance Goals**:
*   Data Loading: Efficient caching/memory mapping for 3D volumes.
*   Training: Support mixed-precision (AMP) for speed.
**Constraints**:
*   Strict separation between Data and Training engines.
*   Reproducibility is paramount (fixed seeds).
*   No hardcoded hyperparameters.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Scientific Reproducibility**: Plan enforces config-based experiments and fixed seeds.
- [x] **Modularity**: Data and Training engines are distinct components.
- [x] **Modern Standards**: Uses PyTorch/MONAI as mandated.
- [x] **Testability**: Includes unit tests and "Quick Test" mode.
- [x] **Clean Architecture**: `models.py`, `dataset_gen.py` separation observed.

## Project Structure

### Documentation (this feature)

```text
specs/001-aneurysm-experiment-pipeline/
├── plan.md              # This file
├── research.md          # Technology choices and architecture decisions
├── data-model.md        # Schema for experiments.json and Artifact definitions
├── quickstart.md        # usage guide for researchers
└── checklists/          # Quality gates
```

### Source Code (repository root)

```text
aneurysm_cnn/
├── data_engine/
│   ├── dataset_gen.py       # Entry point for dataset generation
│   ├── classes.csv          # Data manifest
│   └── src/                 # Preprocessing logic (crop, resize, resample)
├── training_engine/
│   ├── train_models.py      # Entry point for training
│   ├── experiments.json     # Configuration file
│   └── src/
│       ├── models.py        # PyTorch/MONAI Model definitions
│       ├── training.py      # Training loops
│       ├── data_preprocess.py # Augmentation & Loading
│       ├── plots.py         # Artifact generation
│       └── orchestrator.py  # Experiment flow control
├── environment/             # HPC Cluster scripts (CUDA/ROCm)
└── tests/                   # Unit and integration tests
```

**Structure Decision**: Adopts the "Dual Engine" structure defined in the Constitution to separate immutable data generation from iterative model training.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple Dataset Variants (A-E) | Research requires empirical comparison of preprocessing effects. | Single dataset prevents finding optimal input strategy. |
| Dual GPU Support (CUDA/ROCm) | Cluster hardware availability varies. | Vendor lock-in limits compute resource usage. |
