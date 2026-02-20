# Feature List: Aneurysm Experiment Pipeline

This document is the standalone master specification for refactoring and verifying the Aneurysm Experiment Pipeline. It consolidates all requirements, data models, and acceptance criteria to serve as the single source of truth.

## 1. Feature: DICOM Ingestion & NIfTI Standardization
**Goal**: Robustly convert raw DICOM series into standard NIfTI-1 files, handling real-world data issues and ensuring header integrity.

### Requirements
*   **[REQ-1.1]** Convert raw DICOM series (multiple slices) into a single 3D NIfTI (`.nii.gz`) volume.
*   **[REQ-1.2]** Corrupted or incomplete DICOM files MUST be logged as warnings and skipped. The process MUST be resilient and not crash the entire batch.
*   **[REQ-1.3]** Reliably parse and preserve spatial metadata from DICOM tags:
    *   **Pixel Spacing**: (0028,0030) and Slice Thickness (0018,0050).
    *   **Orientation**: Image Orientation Patient (0020,0037).
    *   **Position**: Image Position Patient (0020,0032).
*   **[REQ-1.4]** Output volumes MUST be normalized to Float32 data type.

### Acceptance Criteria
*   **[AC-1.1]** Given a valid DICOM series folder, the tool produces a NIfTI file with an affine matrix that matches the physical coordinates of the source.
*   **[AC-1.2]** When a series contains a corrupted slice, a clear error is logged, the output file is not created (or marked as invalid), and the next series is processed.

---

## 2. Feature: Dataset Variant Generation (A-E)
**Goal**: Generate 5 specific dataset variants to allow empirical comparison of preprocessing effects on model performance.

### Requirements
*   **[REQ-2.1]** Generate the following 5 variants for every case:
    *   **Variant A**: Resampled to 1.0mm isotropic + Cropped to 128x128x128.
    *   **Variant B**: Resampled to 1.0mm isotropic + Shrunk to 128x128x128.
    *   **Variant C**: Cropped only (native resolution) to 128x128x128.
    *   **Variant D**: Shrunk only (native resolution) to 128x128x128.
    *   **Variant E**: Isotropically resampled to largest dimension = 128 + Zero-padded to 128x128x128.
*   **[REQ-2.2]** All resampling operations MUST use **Trilinear interpolation**.
*   **[REQ-2.3]** Volumes smaller than the target crop size (128x128x128) MUST be **Zero-padded** to reach the target dimensions.
*   **[REQ-2.4]** Output datasets MUST follow a strict directory structure:
    ```text
    /path/to/dataset_[VARIANT]/
    ├── 0/ (Negative/No Aneurysm)
    │   └── case_id.nii.gz
    └── 1/ (Positive/Aneurysm)
        └── case_id.nii.gz
    ```

### Acceptance Criteria
*   **[AC-2.1]** Every NIfTI file in Dataset A and B has a header indicating exactly 1.0mm x 1.0mm x 1.0mm spacing.
*   **[AC-2.2]** Every NIfTI file in all datasets has dimensions exactly 128x128x128.

---

## 3. Feature: Experiment Configuration & Orchestration
**Goal**: Implement a robust, automated experiment runner that processes a queue of configurations.

### Requirements
*   **[REQ-3.1]** Define experiment parameters in a `experiments.json` file.
*   **[REQ-3.2]** The system MUST execute the list of experiments sequentially and autonomously.
*   **[REQ-3.3]** Implement schema validation for the configuration file to catch errors (missing fields, invalid types) before execution starts.
*   **[REQ-3.4]** If an experiment fails (e.g., out of memory, script error), the system MUST log the stack trace and proceed to the next experiment in the list.

### Data Model (`experiments.json`)
Each experiment object must include:
*   `name`: Unique identifier.
*   `model`: Architecture key (e.g., "R3D18_MONAI", "R3D50_MONAI").
*   `data_path_key`: Target variant key (e.g., "DATASET_A").
*   `balancing`: Strategy key ("none", "oversampling", "weighted_loss").
*   `augmentation`: Boolean toggle.
*   `aug_probs`: Sub-object with `flip`, `rotate`, `noise` probabilities (0.0 - 1.0).
*   `hold_out_test_set`: Boolean toggle.
*   `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`, `N_SPLITS`.

---

## 4. Feature: Data Loading & Splitting Logic
**Goal**: Implement scientifically rigorous data splitting to ensure reproducibility and prevent leakage.

### Requirements
*   **[REQ-4.1]** Implement **Stratified K-Fold Cross-Validation** to ensure class ratios are preserved across all folds.
*   **[REQ-4.2]** Implement a "Hold-Out Test Set" reservation:
    *   Reserve a fixed percentage (default 20%) of the total data before any CV splitting.
    *   This set is ONLY used for final evaluation if `hold_out_test_set` is enabled.
*   **[REQ-4.3]** Ensure no **Data Leakage**: The split logic MUST be based on unique Patient IDs (if applicable) or Case IDs to prevent samples from the same patient appearing in both training and validation/test sets.
*   **[REQ-4.4]** Use a fixed random seed for all splitting operations to ensure identical splits across different runs with the same config.

### Acceptance Criteria
*   **[AC-4.1]** Sum of samples in Train/Val/Test equals the total number of input samples.
*   **[AC-4.2]** Intersection of file lists between Train, Val, and Test sets is empty for every fold.

---

## 5. Feature: Model Factory & Training Loop
**Goal**: Standardize the training pipeline and ensure model reproducibility.

### Requirements
*   **[REQ-5.1]** Implement a Model Factory supporting 3D ResNet variants (ResNet-18, ResNet-50) using MONAI or PyTorch implementations.
*   **[REQ-5.2]** Fix global random seeds for `random`, `numpy`, `torch`, and `monai` at the start of every experiment.
*   **[REQ-5.3]** Implement a **Quick Test Mode**:
    *   Triggered by a flag or specific config.
    *   Runs only 1 fold for 1 epoch on a small subset of data.
    *   MUST complete in under 5 minutes on reference hardware to verify the end-to-end pipeline.

---

## 6. Feature: Metrics & Artifact Generation
**Goal**: Automatically capture and organize all results for scientific review.

### Requirements
*   **[REQ-6.1]** Primary Metric: The **F2-Score** (weighted to prioritize Recall/Sensitivity) MUST be used for "best model" selection.
*   **[REQ-6.2]** Save the following artifacts for every experiment in `experiments/<name>/`:
    *   `*_model_best.pth`: Weights with the highest validation F2-score.
    *   `*_model_last.pth`: Weights after the final epoch.
    *   `*_metrics.csv`: Per-epoch training/validation loss and metrics.
    *   `*_predictions.csv`: Sample-level output (filename, true label, predicted probability, predicted label).
    *   `*_roc_curve.png`: ROC plot with Area Under Curve (AUC).
    *   `*_cm_raw.png` & `*_cm_normalized.png`: Confusion matrices.
*   **[REQ-6.3]** Generate a `summary_metrics.csv` containing the mean and standard deviation of final metrics across all K-folds.

---

## 7. Feature: Augmentation & Balancing
**Goal**: Provide tools to combat class imbalance and improve model generalization.

### Requirements
*   **[REQ-7.1]** Support on-the-fly 3D augmentations applied ONLY to the training split:
    *   **Random Flip**: Spatial axis 0.
    *   **Random 90-degree Rotation**.
    *   **Random Gaussian Noise**.
*   **[REQ-7.2]** Augmentation probabilities MUST be individually configurable in `experiments.json`.
*   **[REQ-7.3]** Support class balancing strategies:
    *   **Weighted Random Sampler**: Oversamples the minority class during batch creation.
    *   **Weighted Loss**: Applies a higher weight to the minority class in the loss function (e.g., CrossEntropyLoss weights).

---

## 8. Feature: HPC Environment Integration
**Goal**: Ensure the pipeline is portable and runs reliably on local and cluster hardware (NVIDIA/AMD).

### Requirements
*   **[REQ-8.1]** Provide a minimal, pinned `requirements.txt` or `environment.yml` for Python 3.11.
*   **[REQ-8.2]** Provide SLURM batch scripts for both:
    *   **NVIDIA Clusters**: Optimized for CUDA 12.1+.
    *   **AMD Clusters**: Optimized for ROCm 6.0+ (including necessary environment variables like `HSA_OVERRIDE_GFX_VERSION`).
*   **[REQ-8.3]** The system MUST automatically detect and utilize the available GPU (CUDA or ROCm) without manual code changes.
