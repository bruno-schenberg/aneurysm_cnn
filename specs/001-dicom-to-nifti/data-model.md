# Phase 1: Data Model & Entities

The `dicom-to-nifti` feature operates as a data pipeline, meaning it primarily passes data structures between functions rather than persisting entities to a database. 

## Entity 1: `ExamMetadata`
Represents the core information parsed from the DICOM files required for pipeline execution and logging.

**Type**: Python Dictionary

**Fields**:
- `original_name` (string): The raw folder name.
- `fixed_name` (string): The sanitized folder name used for output files.
- `data_path` (string): Relative path to the DICOM series directory.
- `class` (string): The classification label (e.g., '0' or '1').
- `validation_status` (string): Status from earlier pipeline steps (e.g., 'OK').

## Entity 2: `ConversionResult`
Represents the final state of an attempted conversion, used to construct the comprehensive audit log.

**Type**: Python Dictionary

**Fields**:
- `exam_name` (string): The identifier of the exam (`fixed_name`).
- `status` (string): Enum-like value (`success`, `failed`, `skipped`).
- `reason` (string): Detailed explanation if `status` is `failed` or `skipped`. Empty if `success`.
- `output_path` (string): The absolute path to the generated `.nii.gz` file (if successful).

## State Transitions
1. **Pending**: Exam is queued in the `eligible_exams` list.
2. **Checking**: System verifies if `output_path` already exists (Idempotency check).
   - If exists -> transition to **Skipped**.
3. **Processing**: System loads DICOMs via MONAI.
   - If Load/Parse fails -> transition to **Failed** (reason logged).
4. **Saving**: System writes NIfTI via MONAI.
   - If Write fails -> transition to **Failed** (reason logged).
5. **Complete**: NIfTI successfully saved -> transition to **Success**.