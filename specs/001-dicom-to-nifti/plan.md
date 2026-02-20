# Implementation Plan: DICOM Ingestion & NIfTI Standardization

**Branch**: `001-dicom-to-nifti` | **Date**: 2026-02-20 | **Spec**: [Link to Spec](./spec.md)
**Input**: Feature specification from `/specs/001-dicom-to-nifti/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Robustly convert raw DICOM series into standard NIfTI-1 files, handling real-world data issues and ensuring header integrity. The process includes a pre-processing step to analyze raw DICOM directories and compute adjusted/sanitized names for output files. **Critically, the original raw folder names and directory structures MUST NOT be changed on disk.** The sanitized names will only exist in memory (as Python dictionaries) and be recorded in the output logs. We will utilize MONAI to replace the existing `dicom2nifti` pipeline, leveraging its built-in robustness for medical image loading, metadata preservation, and casting to Float32. A comprehensive audit logging system will be integrated to ensure full provenance of the ingestion process without modifying the original raw data.

## Technical Context

**Language/Version**: Python (existing version)
**Primary Dependencies**: MONAI, nibabel, pydicom (existing), python built-in `logging` and `csv`.
**Storage**: Local file system (NIfTI files, log files, CSV summary).
**Testing**: pytest (unit testing for conversion and logging logic).
**Target Platform**: Linux server.
**Project Type**: Data engineering script/pipeline module within `data_engine`.
**Performance Goals**: Process batches sequentially without crashing.
**Constraints**: 
- MUST NOT modify or delete source DICOM files. 
- Process MUST be idempotent (skip existing outputs safely).
- Output must be strictly `Float32`.
**Scale/Scope**: Designed to handle batches of thousands of series, logging corrupted items without halting the overarching process.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Scientific Reproducibility & Rigor**: Pass. Idempotent design ensures consistent outputs. Detailed audit logs allow tracking of every exam's outcome.
- **II. Modularity & Separation of Concerns**: Pass. The new implementation will reside cleanly within `data_engine/src/nifti_utils.py` and potentially a new `logging_utils.py`, keeping the data engine strictly separate from the training engine.
- **III. Modern Technical Standards**: Pass. The pipeline is explicitly adopting MONAI and NiBabel, aligning directly with the approved constitution library selections.
- **IV. Testability & Verification**: Pass. Unit tests will be implemented for the DICOM loading, filtering, NIfTI saving, and log formatting logic.
- **V. Clean Architecture & Readable Code**: Pass. We will use descriptive variable names and ensure all mathematical or medical imaging transformations (like affine parsing) are heavily commented.

## Project Structure

### Documentation (this feature)

```text
specs/001-dicom-to-nifti/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
data_engine/
├── src/
│   ├── class_utils.py
│   ├── dicom_utils.py
│   ├── file_utils.py
│   ├── nifti_utils.py        # Updated to use MONAI and handle robustness
│   ├── logging_utils.py      # New: Centralized audit logging mechanism
│   └── nifti_resize.py
├── data_cleaner.py           # Updated to orchestrate the new logging and sequential processing
└── requirements.txt          # Updated to include MONAI

tests/
└── data_engine/
    └── test_nifti_utils.py   # New: Unit tests for conversion and error handling
```

**Structure Decision**: We will enhance the existing `data_engine` sub-project by modifying `nifti_utils.py` and `data_cleaner.py`. A new module `logging_utils.py` will encapsulate the audit and error logging logic, promoting separation of concerns.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
