# Requirements Quality Checklist: Aneurysm Experiment Pipeline

**Purpose**: Validation of requirements quality for scientific rigor and robustness.
**Created**: 2026-02-18
**Feature**: [Link to spec.md](../spec.md)

## Scientific Rigor & Reproducibility

- [ ] CHK001 Are the random seed requirements explicitly defined for all non-deterministic operations (splitting, initialization, augmentation)? [Completeness, Spec SC-001]
- [ ] CHK002 Is the exact mechanism for "Hold-Out Test Set" reservation (e.g., stratified split before CV) clearly specified to prevent data leakage? [Clarity, Spec US-5]
- [ ] CHK003 Are the 5 dataset variants (A-E) defined with precise geometric parameters (target resolution, dimensions)? [Precision, Spec FR-003]
- [ ] CHK004 Is the "Quick Test" mode defined in a way that guarantees it exercises the *exact* same code paths as full training (just fewer iterations)? [Consistency, Spec FR-007]
- [ ] CHK005 Are the metrics for "best model" selection (e.g., F2-score vs Accuracy) explicitly defined? [Completeness, Gap]

## Data Integrity & Preprocessing

- [ ] CHK006 Is the handling of corrupted DICOM files specified with a verifiable outcome (e.g., "log and skip" vs "abort")? [Edge Case, Spec US-1]
- [ ] CHK007 Are the interpolation methods for resampling (e.g., trilinear vs nearest neighbor) specified? [Precision, Gap]
- [ ] CHK008 Is the behavior defined when an input image is smaller than the target crop size (128x128x128)? [Edge Case, Gap]
- [ ] CHK009 Are the specific augmentation probability parameters and their default ranges documented? [Completeness, Spec FR-014]

## Experimental Configuration & Artifacts

- [ ] CHK010 Are all required fields in `experiments.json` listed with their data types and allowed values? [Completeness, Spec Key Entities]
- [ ] CHK011 Is the format of the output CSVs (columns, precision) defined sufficiently for downstream analysis tools? [Clarity, Spec FR-008]
- [ ] CHK012 Are the supported model architecture strings (e.g., "ResNet18", "R3D-18") explicitly enumerated? [Completeness, Clarifications]
- [ ] CHK013 Is the directory structure for saved artifacts (weights, plots) unambiguously defined? [Clarity, Spec US-3]

## Pipeline Robustness & Environment

- [ ] CHK014 Are the resource requirements (GPU memory) for the different dataset variants and models estimated or constrained? [Non-Functional, Gap]
- [ ] CHK015 Are the specific versions of CUDA and ROCm required for the environment specs documented? [Completeness, Spec FR-011]
- [ ] CHK016 Is the behavior defined if the "Data Engine" output is missing when the "Training Engine" starts? [Exception Flow, Spec Edge Cases]
- [ ] CHK017 Are the failure modes for HPC job submissions (e.g., timeout, preemption) addressed in requirements? [Gap]

## Validation & Verification

- [ ] CHK018 Can the "Verification Speed" target (< 5 mins) be objectively measured on reference hardware? [Measurability, Spec SC-004]
- [ ] CHK019 Are the acceptance criteria for "Artifact Completeness" verifiable via an automated script? [Measurability, Spec SC-003]
