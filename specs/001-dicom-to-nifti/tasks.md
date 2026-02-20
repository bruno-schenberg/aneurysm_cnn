# Implementation Tasks: DICOM Ingestion & NIfTI Standardization

**Feature Branch**: `001-dicom-to-nifti`
**Implementation Plan**: `specs/001-dicom-to-nifti/plan.md`

## Phase 1: Setup
*Goal: Initialize the project with necessary dependencies.*

- [ ] T001 Update dependencies in `data_engine/requirements.txt` to include pinned versions of `monai` and `nibabel` (to ensure scientific reproducibility per Constitution), and ensure `pytest` is present for testing.

## Phase 2: Foundational
*Goal: Implement blocking prerequisites for all user stories.*

- [ ] T002 Implement centralized logging setup (file and stdout) in `data_engine/src/logging_utils.py` to be used across the ingestion pipeline.

## Phase 3: User Story 1 - Process Valid DICOM Series (P1)
*Goal: Convert raw DICOM series into NIfTI volumes using MONAI while preserving metadata and normalizing to Float32, using in-memory name mapping.*
*Independent Test*: Given a valid DICOM directory, verify a `.nii.gz` file is produced with a sanitized name in the output directory, while the source folder remains unchanged.

- [ ] T003 [P] [US1] Create unit tests for happy path conversion and name sanitization in `data_engine/tests/test_nifti_utils.py`.
- [ ] T004 [US1] Implement in-memory directory analysis and folder-name-to-sanitized-name mapping in `data_engine/src/file_utils.py`.
- [ ] T005 [US1] Implement `convert_series_to_nifti` using MONAI `LoadImage` and `SaveImage` transforms in `data_engine/src/nifti_utils.py`.
- [ ] T006 [US1] Implement `process_and_convert_exams` in `data_engine/src/nifti_utils.py` to handle batch iteration and skip existing outputs (idempotency), using the name mapping.
- [ ] T007 [US1] Update `data_engine/data_cleaner.py` to orchestrate the new analysis and conversion steps without modifying source disk structure.

## Phase 4: User Story 2 - Resilient Handling of Corrupted DICOM Data (P1)
*Goal: Prevent pipeline crashes by catching corrupted slices/files, logging them as warnings, and continuing.*
*Independent Test*: Given a batch with a corrupted DICOM slice, verify the corrupted series is skipped, a warning is logged, and the rest of the batch completes.

- [ ] T008 [P] [US2] Add unit tests for handling corrupted series (e.g., missing spatial tags, invalid format) in `data_engine/tests/test_nifti_utils.py`.
- [ ] T009 [US2] Update `convert_series_to_nifti` in `data_engine/src/nifti_utils.py` to catch specific MONAI/pydicom exceptions.
- [ ] T010 [US2] Update `process_and_convert_exams` in `data_engine/src/nifti_utils.py` to log caught errors using `logging_utils.py` and continue to the next series.

## Phase 5: User Story 3 - Comprehensive Processing Audit Log (P2)
*Goal: Output a complete summary CSV log containing the outcome of every processed exam.*
*Independent Test*: Verify `ingestion_summary.csv` matches the total number of input exams and correctly maps success/failure statuses and reasons.

- [ ] T011 [P] [US3] Add unit tests for `ConversionResult` dictionary generation and CSV formatting in `data_engine/tests/test_nifti_utils.py`.
- [ ] T012 [US3] Refactor `process_and_convert_exams` in `data_engine/src/nifti_utils.py` to aggregate and return a list of `ConversionResult` objects (including sanitized names).
- [ ] T013 [US3] Update `data_engine/data_cleaner.py` to write the returned `ConversionResult` list to `ingestion_summary.csv` using Python's `csv` module.

## Phase 6: Polish
*Goal: Address cross-cutting concerns, clean code, and verify end-to-end functionality.*

- [ ] T014 Update type hints, docstrings, and inline comments in `data_engine/src/nifti_utils.py` and `data_engine/data_cleaner.py` to adhere to constitution standards.
- [ ] T015 Run full pipeline sanity check via `data_engine/data_cleaner.py` using a representative data sample, including a specific test against a read-only source directory, to verify zero modifications to raw data.

---

## Dependencies

- **US1** requires Foundational setup (T001, T002) and name mapping (T004) to be completed.
- **US2** builds upon the conversion logic established in US1.
- **US3** modifies the reporting structure established in US2.

## Parallel Execution Examples

- **Test-Driven Parallelism**: Tasks T003 [US1], T008 [US2], and T011 [US3] can be started in parallel while core logic is being implemented.

## Implementation Strategy

1. **Analysis & Mapping**: First, ensure the raw data can be correctly analyzed and mapped to sanitized names in memory.
2. **MVP Conversion**: Deliver the core happy path NIfTI conversion using the new MONAI integration.
3. **Resilience & Audit**: Layer on error handling and the final audit log.
