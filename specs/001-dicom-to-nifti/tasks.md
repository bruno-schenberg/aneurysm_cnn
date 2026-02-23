---
description: "Task list for DICOM Ingestion & NIfTI Standardization implementation"
---

# Tasks: DICOM Ingestion & NIfTI Standardization

**Input**: Design documents from `/specs/001-dicom-to-nifti/`
**Prerequisites**: plan.md, spec.md, data-model.md, research.md, quickstart.md, contracts/cli.md

## Phase 1: Setup

**Purpose**: Project initialization and basic structure

- [ ] T001 Update dependencies by adding `monai` to `data_engine/requirements.txt`
- [ ] T002 [P] Create new file `data_engine/src/logging_utils.py`
- [ ] T003 [P] Create empty test files: `tests/data_engine/test_file_utils.py`, `tests/data_engine/test_dicom_utils.py`, `tests/data_engine/test_nifti_utils.py`

---

## Phase 2: Foundational

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [ ] T004 Implement comprehensive audit and error logging mechanism (JSONL and CSV support) in `data_engine/src/logging_utils.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Process Valid DICOM Series (Priority: P1) 🎯 MVP

**Goal**: Convert a folder containing a valid DICOM series into a single 3D NIfTI volume with preserved spatial metadata and Float32 normalization.

**Independent Test**: Can be tested by providing a known-good DICOM series directory and verifying the output `.nii.gz` file's dimensions, data type, and affine matrix.

### Tests for User Story 1 ⚠️

- [ ] T005 [P] [US1] Write unit tests for standardized naming and folder stats in `tests/data_engine/test_file_utils.py`
- [ ] T006 [P] [US1] Write unit tests for basic DICOM loading and metadata extraction in `tests/data_engine/test_dicom_utils.py`
- [ ] T007 [P] [US1] Write unit tests for NIfTI conversion and spatial metadata preservation in `tests/data_engine/test_nifti_utils.py`

### Implementation for User Story 1

- [ ] T008 [P] [US1] Update `data_engine/src/file_utils.py` to standardize case names (e.g., BP001) and assign data codes
- [ ] T009 [P] [US1] Update `data_engine/src/dicom_utils.py` to load DICOM metadata, extract orientation, spatial tags, and exam size
- [ ] T010 [US1] Update `data_engine/src/nifti_utils.py` to implement `convert_series_to_nifti` using MONAI for Float32 normalization and affine preservation
- [ ] T011 [US1] Update `data_engine/data_cleaner.py` to orchestrate valid DICOM conversion using the updated utils

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Resilient Handling of Corrupted DICOM Data (Priority: P1)

**Goal**: Gracefully handle corrupted or incomplete DICOM slices, missing spatial tags, mixed series, and scout images without crashing.

**Independent Test**: Can be tested by placing a corrupted or non-DICOM file inside a series folder within a batch and verifying the batch completes processing other series.

### Tests for User Story 2 ⚠️

- [ ] T012 [P] [US2] Write unit tests for filtering, projection sorting, and outlier detection in `tests/data_engine/test_dicom_utils.py`
- [ ] T013 [P] [US2] Write unit tests for error handling and idempotency in `tests/data_engine/test_nifti_utils.py`

### Implementation for User Story 2

- [ ] T014 [P] [US2] Update `data_engine/src/dicom_utils.py` to filter scout images ('DERIVED'), validate single SeriesInstanceUID, check mandatory spatial tags, and analyze spacing
- [ ] T015 [P] [US2] Update `data_engine/src/dicom_utils.py` to implement projection-based spatial sorting
- [ ] T016 [US2] Update `data_engine/src/nifti_utils.py` to catch MONAI loading/saving exceptions and skip existing outputs (idempotency check)
- [ ] T017 [US2] Update `data_engine/data_cleaner.py` to catch exceptions from conversion, log specific error codes, and continue sequential processing

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Comprehensive Processing Audit Log (Priority: P2)

**Goal**: Generate a complete summary log at the end of a batch run that lists every processed exam, conversion status, and failure reason.

**Independent Test**: Can be tested by providing a known mix of valid and corrupted series and verifying the summary log accounts for 100% of the input folders with correct statuses.

### Implementation for User Story 3

- [ ] T018 [US3] Update `data_engine/data_cleaner.py` to track `ConversionResult` entities for all processed exams
- [ ] T019 [US3] Update `data_engine/data_cleaner.py` to write final audit log using `logging_utils.py` to `ingestion_summary.csv`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T020 Run entire pipeline (data_cleaner.py) to validate end-to-end integration and error handling
- [ ] T021 Code cleanup and architectural review to ensure clean separation of concerns

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User Story 1 (P1) -> User Story 2 (P1) -> User Story 3 (P2)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### Parallel Opportunities

- Setup tasks (T002, T003) can run in parallel
- Unit tests for US1 (T005, T006, T007) can be written in parallel
- Utils updates for US1 (T008, T009) can run in parallel
- Unit tests for US2 (T012, T013) can be written in parallel
- Utilities updates for US2 (T014, T015) can run in parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently using quickstart instructions

### Incremental Delivery

1. Complete Setup + Foundational
2. Add User Story 1 → Test independently
3. Add User Story 2 → Test independently
4. Add User Story 3 → Test independently
