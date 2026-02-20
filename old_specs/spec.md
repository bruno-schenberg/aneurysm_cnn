# Feature Specification: Aneurysm Experiment Pipeline

**Feature Branch**: `001-aneurysm-experiment-pipeline`  
**Created**: 2026-02-18  
**Status**: Draft  
**Input**: User description: "This application is to run experiments with Computer Vision models to evaluate their performance on the classification task of aneurysms in angio-magnetic ressonance brain exams. Each experiment is defined in a config file, including dataset, model, hyperparameters. Data_engine is responsible by processing the raw dicom files, generating nifti files, creating datasets that differ on the crop/shrink strategy, and the use (or not use) of isotropic resampling. Training engine is responsible by running all experiments in the list defined in a config file, performing k-fold cross validation, and generating all the artifacts after each experiment is done. Artifacts include the final model weights, the best weights, confusion matrixes, roc curve, and other metrics. There must be support for quick tests (only one fold) and other unit tests to verify that each step of the pipeline works as intended. The code must be well documented and explainable to outside viewers and examiners."

## Clarifications

### Session 2026-02-18
- Q: Which specific 3D architectures must be supported for the initial implementation? -> A: 3D ResNet variants (e.g., ResNet-18, ResNet-50) only.
- Q: How should class imbalance be handled? -> A: Support both Weighted Random Sampler and Weighted Loss, configurable via the JSON file.
- Q: How should the size of the test split be defined? -> A: A fixed hardcoded percentage (e.g., 20%) for all experiments to ensure consistency.
- Q: How should the 'Hold-Out Test Set' be reserved? -> A: Stratified split on the full dataset first to isolate the test set, preserving class ratios.
- Q: Which metric determines the 'best' model checkpoint? -> A: F2-Score (prioritizing recall).
- Q: How is 'Quick Test' mode defined? -> A: Run 1 fold for 1 epoch on a small subset (e.g., 2 batches) of data.
- Q: What is the target isotropic resolution? -> A: 1.0mm isotropic for A/B; Max dimension resize + pad for E.
- Q: Which interpolation method should be used? -> A: Trilinear interpolation.
- Q: How to handle small input images? -> A: Zero pad to target size.
- Q: How to handle corrupted DICOM files? -> A: Log error and skip file.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Dataset Generation (Priority: P1)

As a researcher, I want to process raw DICOM files into training-ready NIfTI datasets with 5 specific preprocessing strategies (combinations of resampling, cropping, shrinking, and padding) so that I can experimentally determine the optimal input format for the model.

**Why this priority**: Essential prerequisite. No training can occur without prepared data.

**Independent Test**: Run `data_engine/dataset_gen.py` on a small subset of DICOMs. Verify that 5 distinct output directories (Dataset A through E) are created and contain NIfTI files with the correct spatial properties (e.g., check header for spacing/dimensions).

**Acceptance Scenarios**:

1. **Given** a raw DICOM input, **When** the data engine runs, **Then** it produces **Dataset A** (Resampled to 1mm + Cropped to 128x128x128).
2. **Given** a raw DICOM input, **When** the data engine runs, **Then** it produces **Dataset B** (Resampled to 1mm + Shrunk to 128x128x128).
3. **Given** a raw DICOM input, **When** the data engine runs, **Then** it produces **Dataset C** (Cropped only to 128x128x128).
4. **Given** a raw DICOM input, **When** the data engine runs, **Then** it produces **Dataset D** (Shrunk only to 128x128x128).
5. **Given** a raw DICOM input, **When** the data engine runs, **Then** it produces **Dataset E** (Isotropically Resampled to largest dim=128 + Padded to 128x128x128).
6. **Given** corrupted DICOM files, **When** the data engine runs, **Then** it handles them gracefully (logs warning/skips) without crashing the entire batch.

---

### User Story 2 - Experiment Execution (Priority: P1)

As a researcher, I want to define multiple experiments in a single JSON configuration file (specifying dataset, model, hyperparameters) and have the system execute them sequentially so that I can compare different approaches automatedly.

**Why this priority**: Core functionality of the "Training Engine". Enables the research workflow.

**Independent Test**: Create a config file with 2 different minimal experiments. Run the training engine. Verify that 2 distinct output directories are created with training logs.

**Acceptance Scenarios**:

1. **Given** a `experiments.json` file with Experiment A (Model X) and Experiment B (Model Y), **When** the training engine is executed, **Then** it runs Experiment A followed by Experiment B.
2. **Given** an experiment config with specific hyperparameters (e.g., learning rate), **When** training starts, **Then** the model initializes and trains using those specific parameters.

---

### User Story 3 - Cross-Validation & Artifacts (Priority: P2)

As a researcher, I want the system to automatically perform k-fold cross-validation and generate comprehensive artifacts (weights, ROC curves, confusion matrices, CSV logs) for each experiment so that I can statistically validate model performance and analyze detailed results.

**Why this priority**: Critical for scientific validity (reproducibility and rigorous evaluation).

**Independent Test**: Run a short experiment with `k=2`. Verify that 2 separate validation folds occurred and that the specific artifact files listed below are generated for each fold.

**Acceptance Scenarios**:

1. **Given** an experiment configured with `k=5` folds, **When** the experiment completes, **Then** there are results for 5 distinct folds and one aggregated performance report.
2. **Given** a completed training run for a specific fold, **When** I check the output folder, **Then** I find the following files (where `*` matches the experiment and fold ID):
    - `*_model_best.pth` (Weights with best validation metric)
    - `*_model_last.pth` (Weights from final epoch)
    - `*_cm_raw.png` (Raw Confusion Matrix plot)
    - `*_cm_normalized.png` (Normalized Confusion Matrix plot)
    - `*_roc_curve.png` (ROC Curve plot)
    - `*_predictions.csv` (Per-sample predictions: filename, true_label, prediction)
    - `*_metrics.csv` (Training history + Final Precision/Recall/F2-Score)

---

### User Story 4 - Pipeline Verification (Priority: P3)

As a developer/researcher, I want a "quick test" mode and unit tests so that I can verify the pipeline works as intended without waiting for full training runs.

**Why this priority**: Accelerates development and debugging loop.

**Independent Test**: Run the training script with a `--quick` flag. Verify it completes a single epoch/fold cycle in under 2 minutes (assuming small data).

**Acceptance Scenarios**:

1. **Given** the flag `--quick` (or config equivalent), **When** the training engine runs, **Then** it executes only one fold and a reduced number of epochs/steps to verify end-to-end flow.
2. **Given** unit tests for the pipeline, **When** I run `pytest` (or equivalent), **Then** all core data transformation and metric calculation logic passes.

---

### User Story 5 - Hold-Out Test Set Evaluation (Priority: P2)

As a researcher, I want to reserve a portion of the data as a hold-out test set and only evaluate the model on it when explicitly configured, so that I can perform a final unbiased assessment of the model's performance.

**Why this priority**: Prevents overfitting to the validation set during hyperparameter tuning.

**Independent Test**: Configure an experiment with `hold_out_test_set: true`. Check logs/output to see a final evaluation phase on the test split after cross-validation or training completes.

**Acceptance Scenarios**:

1. **Given** an experiment config with `hold_out_test_set: true`, **When** the experiment completes, **Then** the system evaluates the best model on the unseen test set and saves the results.
2. **Given** an experiment config with `hold_out_test_set: false` (default), **When** the experiment completes, **Then** no test set evaluation is performed (only validation results are reported).
3. **Given** a dataset, **When** the Data Engine runs or splits are generated, **Then** a dedicated test split is created using stratified sampling and excluded from the training/validation folds.

---

### User Story 6 - Environment & HPC Execution (Priority: P2)

As a researcher, I want to run the training pipeline on High Performance Computing (HPC) clusters using either NVIDIA (CUDA) or AMD (ROCm) GPUs, so that I can utilize available hardware resources efficiently.

**Why this priority**: Research often requires significant compute resources available only on clusters with specific GPU architectures.

**Independent Test**: Submit a job using `sbatch` with the provided SLURM script (modified for a test queue) and verify it activates the correct Conda environment, detects the GPU, and runs the training script.

**Acceptance Scenarios**:

1. **Given** an NVIDIA GPU environment, **When** I run the `environment/cuda/start_script_cuda.slurm`, **Then** the system activates the `aneurysm_cnn_cuda` environment and executes the training using CUDA.
2. **Given** an AMD GPU environment, **When** I run the `environment/rocm/start_script_rocm.slurm`, **Then** the system activates the `aneurysm_cnn_rocm` environment, sets `HSA_OVERRIDE_GFX_VERSION` if needed, and executes the training using ROCm.
3. **Given** a local execution context, **When** I execute the training script directly (after conda activation), **Then** it runs successfully without requiring SLURM (provided dependencies are met).

---

### User Story 7 - On-the-Fly Data Augmentation (Priority: P2)

As a researcher, I want to apply random 3D data augmentations (flip, rotation, noise) during the training phase on-the-fly and configure their intensity/probability per experiment, so that I can improve model generalization and robustness to variations without exploding storage requirements.

**Why this priority**: Standard practice in medical imaging deep learning to prevent overfitting on small datasets.

**Independent Test**: Configure an experiment with `augmentation: true` and high noise probability. Visualize or log a batch of training images to verify they differ from the original static files and change across epochs.

**Acceptance Scenarios**:

1. **Given** an experiment config with `augmentation: true`, **When** the training loader fetches a batch, **Then** random transformations (Flip, Rotate90, Gaussian Noise) are applied to the tensor before feeding the model.
2. **Given** an experiment config with `augmentation: false`, **When** the training loader fetches a batch, **Then** only deterministic transforms (Load, Resize, ScaleIntensity) are applied.
3. **Given** specific probabilities in config (e.g., `aug_flip_prob: 0.0`, `aug_noise_prob: 1.0`), **When** training runs, **Then** the pipeline respects these settings (e.g., no flipping, always adding noise).

### Edge Cases

- **Empty/Invalid Config**: What happens if `experiments.json` is missing required fields? (Should validate schema and fail early).
- **Data Mismatch**: What happens if an experiment specifies a dataset key that wasn't generated by the Data Engine? (Should error out clearly).
- **Resource Exhaustion**: How does the system handle running out of GPU memory during a batch of experiments? (Ideally fail the current experiment and try the next, or fail safely).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support defining experiments via a JSON configuration file including fields for `dataset`, `model_architecture` (must support 3D ResNet-18/50 variants), `hyperparameters`, and `balancing_strategy` (supporting 'Weighted Sampler', 'Weighted Loss', or 'Both').
- **FR-002**: The Data Engine MUST convert raw DICOM series to NIfTI format. Corrupted files MUST be logged and skipped.
- **FR-003**: The Data Engine MUST generate 5 distinct dataset variants:
    - **Dataset A**: Resampled to 1.0mm isotropic + Cropped to 128³
    - **Dataset B**: Resampled to 1.0mm isotropic + Shrunk to 128³
    - **Dataset C**: Cropped only to 128³
    - **Dataset D**: Shrunk only to 128³
    - **Dataset E**: Resized (largest dim=128, then isotropic) + Padded to 128³
- **FR-004**: The Training Engine MUST implement k-fold cross-validation (user-configurable `k`).
- **FR-005**: The system MUST save model weights: `*_model_best.pth` (best Validation F2-Score) and `*_model_last.pth` (final epoch).
- **FR-006**: The system MUST generate evaluation plots: Raw Confusion Matrix (`*_cm_raw.png`), Normalized Confusion Matrix (`*_cm_normalized.png`), and ROC Curve (`*_roc_curve.png`).
- **FR-007**: The system MUST support a "Quick Test" mode that runs a minimal version of the pipeline (e.g., 1 fold, 1 epoch, subset of data) for verification.
- **FR-008**: The system MUST log detailed metrics to CSV:
    - `*_metrics.csv`: Contains epoch-by-epoch training/eval metrics and a final row with Precision, Recall, and F2-Score.
    - `*_predictions.csv`: Contains filename, true label, and predicted label for every sample in the validation set.
- **FR-009**: The system MUST support an optional "Hold-Out Test Set" configuration flag. When enabled, the Training Engine MUST reserve a fixed percentage (e.g., 20%) of data for final testing and NOT use it for training or cross-validation.
- **FR-010**: The system MUST generate an aggregated summary CSV containing average Precision, Recall, F2-Score, Accuracy, and Loss across all folds.
- **FR-011**: The system MUST provide reproducible Conda environment specifications (`environment.yml`) and SLURM batch scripts for both NVIDIA (CUDA 12.1+) and AMD (ROCm 6.0+) GPU clusters.
- **FR-012**: The Training Engine MUST support on-the-fly 3D data augmentation using MONAI transforms, applied only to the training split (not validation/test).
- **FR-013**: The supported augmentations MUST include:
    - Random Flip (spatial axis 0)
    - Random 90-degree Rotation
    - Random Gaussian Noise
- **FR-014**: Augmentation usage and specific probabilities (e.g., `prob_flip`, `prob_rotate`, `prob_noise`) MUST be configurable per experiment in `experiments.json`.
- **FR-015**: Image resampling MUST use Trilinear interpolation. Images smaller than target crop size MUST be zero-padded.

### Key Entities

- **ExperimentConfig**: JSON object defining the parameters for a single training run.
- **DatasetRegistry**: Mapping between dataset keys (in config) and physical paths/preprocessing types.
- **ArtifactBundle**: The collection of outputs (weights, logs, plots) generated by a single experiment.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: **Reproducibility**: Two runs with the same configuration and random seed MUST produce identical metrics and model weights.
- **SC-002**: **Automation**: A user can define 5 experiments in a config file and the system completes all 5 sequentially without manual intervention.
- **SC-003**: **Artifact Completeness**: 100% of completed experiments MUST have a corresponding ROC curve, Confusion Matrix, and saved model weights in their output directory.
- **SC-004**: **Verification Speed**: The "Quick Test" mode MUST complete an end-to-end pipeline verification (data load -> train step -> val step -> artifact gen) in under 5 minutes on standard hardware.
