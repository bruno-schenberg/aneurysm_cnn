# Phase 0: Outline & Research

## Decision 1: Library for DICOM to NIfTI Conversion
**Decision**: We will use `MONAI` (Medical Open Network for AI).
**Rationale**: `MONAI` provides robust, production-ready transforms (`LoadImage`, `SaveImage`) specifically designed for medical deep learning pipelines. It handles spatial normalization, orientation preservation, and data type casting (to `Float32`) out-of-the-box, natively supporting both NIfTI and DICOM directory inputs. It is already an approved library in the project Constitution (Principle III).
**Alternatives considered**: 
- `pydicom` + `dicom2nifti` (current implementation): While simple, `dicom2nifti` can be inflexible when forcing specific data types or gracefully skipping single corrupted slices mid-series without failing the entire conversion process.
- `pydicom` + `nibabel` (custom implementation): Offers maximum control but requires writing complex 3D math to manually construct the affine matrix from DICOM spatial tags (`ImageOrientationPatient`, `ImagePositionPatient`, etc.), which increases technical debt and bug risk.

## Decision 2: Audit Logging Mechanism
**Decision**: Use Python's built-in `logging` module and the built-in `csv` module.
**Rationale**: The `logging` module is the industry standard for robust error and warning tracking. It can easily route logs to both stdout and a rolling `.log` file simultaneously. The `csv` module allows us to satisfy the FR-009 requirement of outputting a comprehensive, structured summary log of all processed exams without adding third-party dependencies.
**Alternatives considered**: 
- Structured JSON logging (e.g., `structlog`): Good for Elasticsearch/Kibana ingestion, but overkill for local CLI tools where data engineers need easily readable, columnar data (CSV).
- Simple `print` statements: Fails to meet the persistency and formatting requirements for formal audit trails.

## Decision 3: Idempotency Strategy
**Decision**: Before converting a DICOM series, the system will check if the target NIfTI output file already exists. If it does, the conversion is skipped, and the audit log records "skipped (already exists)". 
**Rationale**: This fulfills the non-destructive (TC-001) and idempotent (TC-002) constraints. It ensures rerunning the batch process only processes new or previously failed exams, drastically saving compute time.
**Alternatives considered**:
- Checksumming source DICOMs and NIfTI files: More rigorous but extremely slow due to the I/O cost of hashing thousands of large 3D medical images. Simple file existence is sufficient for this ingestion pipeline.