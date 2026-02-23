# Implementation Plan: DICOM Ingestion & NIfTI Standardization

**Branch**: `001-dicom-to-nifti` | **Date**: 2026-02-21 | **Spec**: [Link to Spec](./spec.md)
**Input**: Feature specification from `/specs/001-dicom-to-nifti/spec.md`

## Summary

Robustly convert raw DICOM series into standard NIfTI-1 files, handling real-world data issues and ensuring header integrity. The process involves pre-processing and organizing raw DICOM directories (in `file_utils.py`) to generate standardized names (e.g., `BPXXX`), detecting missing cases, and categorizing folders with data codes. Series integrity is validated (in `dicom_utils.py`) including projection-based spatial sorting to handle missing instance numbers, filtering out scout/localizer ('DERIVED') images with warnings, verifying a single `SeriesInstanceUID` per folder (failing on mixed series), verifying mandatory spatial tags, spacing analysis, and anomaly detection. We will utilize MONAI and NiBabel for robust loading, ensuring spatial metadata is preserved correctly in the affine matrix, and normalizing the output to Float32. A comprehensive audit logging system captures success and failure reasons for every processed exam without modifying the original raw data. Output files will be named using the standardized case names to ensure traceability. Existing NIfTI files will be skipped to maintain idempotency.

## Technical Context

**Language/Version**: Python (existing version)
**Primary Dependencies**: MONAI, nibabel, pydicom (existing), numpy, python built-in `logging` and `csv`.
**Storage**: Local file system (NIfTI files, log files, CSV summary).
**Testing**: pytest (unit testing for conversion, filtering, sorting, and logging logic).
**Target Platform**: Linux server.
**Project Type**: Data engineering script/pipeline module within `data_engine`.
**Performance Goals**: Process batches sequentially of thousands of series without crashing.
**Constraints**: 
- MUST NOT modify or delete source DICOM files. 
- Process MUST be idempotent (skip existing outputs safely).
- Output must be strictly `Float32` and properly named (e.g. `BP001.nii.gz`).
**Scale/Scope**: Designed to handle batches of thousands of series, explicitly logging corrupted items, missing spatial tags, and mixed series without halting the overarching process.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Scientific Reproducibility & Rigor**: Pass. Idempotent design, deterministic file naming, projection sorting, and outlier filtering. Detailed audit logs allow tracking of every exam's outcome.
- **II. Modularity & Separation of Concerns**: Pass. Responsibilities are split between `file_utils.py` (organization), `dicom_utils.py` (validation), `nifti_utils.py` (conversion), and potentially a new `logging_utils.py`.
- **III. Modern Technical Standards**: Pass. The pipeline leverages MONAI, NiBabel, and pydicom for robust medical imaging operations.
- **IV. Testability & Verification**: Pass. Unit tests will be implemented for DICOM loading, filtering (scout image removal, mixed series handling), projection sorting logic, NIfTI saving, and log formatting.
- **V. Clean Architecture & Readable Code**: Pass. Descriptive variable names will be used. Explicit handling of complex cases (like projection-based sorting) will be heavily documented.

## Project Structure

### Documentation (this feature)

```text
specs/001-dicom-to-nifti/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
data_engine/
├── src/
│   ├── class_utils.py
│   ├── dicom_utils.py        # Updated to filter scouts, handle mixed series, spatial tags validation
│   ├── file_utils.py         # Updated to provide standardized names and handle duplicate folders
│   ├── nifti_utils.py        # Updated to use MONAI and handle robustness and Float32 type
│   ├── logging_utils.py      # New: Centralized audit logging mechanism
│   └── nifti_resize.py
├── data_cleaner.py           # Updated to orchestrate the new logging, sequential processing, and skip logic
└── requirements.txt          # Updated to include MONAI

tests/
└── data_engine/
    ├── test_file_utils.py    # New: Unit tests for naming and folder stats
    ├── test_dicom_utils.py   # New: Unit tests for filtering, projection sorting, outlier detection
    └── test_nifti_utils.py   # New: Unit tests for conversion, idempotency, and error handling
```

**Structure Decision**: Enhance the existing `data_engine` sub-project by modifying `nifti_utils.py`, `dicom_utils.py`, `file_utils.py`, and `data_cleaner.py`. A new module `logging_utils.py` will encapsulate the audit and error logging logic, promoting separation of concerns.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
