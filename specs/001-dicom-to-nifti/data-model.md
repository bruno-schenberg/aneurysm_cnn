# Phase 1: Data Model & Entities

The `dicom-to-nifti` feature operates as a data pipeline, meaning it primarily passes data structures between functions rather than persisting entities to a database. 

## Entity 1: `ExamMetadata`
Represents the core information parsed from the DICOM files required for pipeline execution and logging.

**Type**: Python Dictionary

**Fields**:
- `original_name` (string): The raw folder name.
- `fixed_name` (string): The sanitized folder name used for output files (e.g., `BP001`).
- `data_code` (string): The categorization of the folder contents (e.g., `READY`, `EMPTY`, `SUBFOLDER_PATH`, `DUPLICATE_DATA`, `MISSING`).
- `data_path` (string): Relative path to the DICOM series directory.
- `class` (string): The classification label (e.g., '0' or '1').
- `validation_status` (string): Status from earlier pipeline steps (e.g., 'OK', 'MIXED_SERIES_ERROR', 'MISSING_SPATIAL_METADATA_ERROR').
- `total_dcms` (int): Number of dicom files.
- `scout_slice_count` (int): Number of derived/scout images filtered out.
- `orientation` (string): 'AXIAL', 'CORONAL', 'SAGITTAL', or 'OBLIQUE'.
- `slice_thickness` (float/string): Calculated effective slice thickness or 'N/A'.
- `exam_size` (float/string): Total physical coverage size or 'N/A'.
- `duplicate_slice_count` (int): Number of duplicate slices found.
- `modality` (string): e.g. 'CT', 'MR', or 'N/A'.
- `patient_sex` (string): Patient sex from DICOM or 'N/A'.
- `image_dimensions` (string): e.g. '512x512'.

## Entity 2: `ConversionResult`
Represents the final state of an attempted conversion, used to construct the comprehensive audit log.

**Type**: Python Dictionary

**Fields**:
- `exam_name` (string): The identifier of the exam (`fixed_name`).
- `status` (string): Enum-like value (`success`, `failed`, `skipped`).
- `reason` (string): Detailed explanation if `status` is `failed` or `skipped`. Empty if `success`.
- `output_path` (string): The absolute path to the generated `.nii.gz` file (if successful).

## State Transitions
1. **Organize**: System standardizes names, detects missing cases, and assigns data codes.
2. **Validate**: System filters scouts, sorts via projection, checks spacing, checks mixed series, and flags outliers.
   - If invalid -> validation_status set to error code.
3. **Pending**: Exam is queued in the `eligible_exams` list.
4. **Checking**: System verifies if `output_path` already exists (Idempotency check).
   - If exists -> transition to **Skipped**.
5. **Processing**: System loads DICOMs via MONAI.
   - If Load/Parse fails -> transition to **Failed** (reason logged).
6. **Saving**: System writes NIfTI via MONAI.
   - If Write fails -> transition to **Failed** (reason logged).
7. **Complete**: NIfTI successfully saved -> transition to **Success**.
