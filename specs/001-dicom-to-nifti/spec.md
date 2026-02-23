# Feature Specification: DICOM Ingestion & NIfTI Standardization

**Feature Branch**: `001-dicom-to-nifti`  
**Created**: 2026-02-20  
**Status**: Draft  
**Input**: User description: "create a new feature branch number 1. ## 1. Feature: DICOM Ingestion & NIfTI Standardization **Goal**: Robustly convert raw DICOM series into standard NIfTI-1 files, handling real-world data issues and ensuring header integrity. ### Requirements * **[REQ-1.1]** Convert raw DICOM series (multiple slices) into a single 3D NIfTI (`.nii.gz`) volume. * **[REQ-1.2]** Corrupted or incomplete DICOM files MUST be logged as warnings and skipped. The process MUST be resilient and not crash the entire batch. * **[REQ-1.3]** Reliably parse and preserve spatial metadata from DICOM tags: * **Pixel Spacing**: (0028,0030) and Slice Thickness (0018,0050). * **Orientation**: Image Orientation Patient (0020,0037). * **Position**: Image Position Patient (0020,0032). * **[REQ-1.4]** Output volumes MUST be normalized to Float32 data type. ### Acceptance Criteria * **[AC-1.1]** Given a valid DICOM series folder, the tool produces a NIfTI file with an affine matrix that matches the physical coordinates of the source to a precision of 4 decimal places. * **[AC-1.2]** When a series contains a corrupted slice, a clear error is logged, the output file is not created (or marked as invalid), and the next series is processed."

## Clarifications

### Session 2026-02-20
- Q: How should batch processing handle multiple series? → A: Sequentially (simpler, lower memory footprint)
- Q: How should the tool discover DICOM series in a batch? → A: Recursively search all subdirectories for valid DICOM series
- Q: How should error logging be handled? → A: Python `logging` module to stdout and a JSONL (JSON lines) file (e.g., `ingestion.log`)
- Q: If the destination NIfTI file already exists, what should the tool do to satisfy the idempotency requirement? → A: Skip conversion and log as already processed
- Q: How should the system handle a folder containing multiple distinct DICOM series intermixed? → A: Fail the entire folder, log 'MIXED_SERIES_ERROR', and skip
- Q: How should the system handle a DICOM series that is missing mandatory spatial tags (e.g., Pixel Spacing)? → A: Log an error and skip the series (treat as corrupted)
- Q: How should the system handle scout or localizer images found within a series folder? → A: Automatically filter out 'DERIVED' images and convert only 'ORIGINAL' ones
- Q: How should the output NIfTI files be named to ensure they are easily linkable to the source exams? → A: Use the standardized case name (e.g., `BP001.nii.gz`)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Process Valid DICOM Series (Priority: P1)

As a data engineer, I need to convert a folder containing a valid DICOM series into a single 3D NIfTI volume with preserved spatial metadata and Float32 normalization, so that the data is ready for downstream machine learning pipelines.

**Why this priority**: This is the core happy-path functionality of the feature. Without this, the ingestion pipeline cannot function.

**Independent Test**: Can be tested by providing a known-good DICOM series directory and verifying the output `.nii.gz` file's dimensions, data type, and affine matrix.

**Acceptance Scenarios**:

1. **Given** a directory containing a valid, complete DICOM series for a single scan, **When** the conversion tool processes the directory, **Then** a single `.nii.gz` file is produced.
2. **Given** the generated `.nii.gz` file, **When** its header is inspected, **Then** the affine matrix correctly reflects the original DICOM's Pixel Spacing, Slice Thickness, Image Orientation Patient, and Image Position Patient.
3. **Given** the generated `.nii.gz` file, **When** its data is loaded, **Then** the voxel values are of type Float32.

---

### User Story 2 - Resilient Handling of Corrupted DICOM Data (Priority: P1)

As a data engineer, I need the ingestion process to gracefully handle corrupted or incomplete DICOM slices by logging a warning, skipping the corrupted series, and continuing with the rest of the batch, so that unattended bulk processing does not crash.

**Why this priority**: Real-world medical data is frequently messy. A pipeline that crashes on the first error is unusable for large datasets.

**Independent Test**: Can be tested by placing a corrupted or non-DICOM file inside a series folder within a batch and verifying the batch completes processing other series.

**Acceptance Scenarios**:

1. **Given** a batch of DICOM series folders where one folder contains a corrupted slice, **When** the batch is processed, **Then** a clear warning/error is logged specifically identifying the corrupted series.
2. **Given** the corrupted series from the previous scenario, **When** the processing finishes, **Then** no NIfTI file is generated for that specific series (or it is clearly marked as invalid).
3. **Given** the batch processing run from the first scenario, **When** the run completes, **Then** all other valid DICOM series in the batch are successfully converted to NIfTI files without the process crashing.

---

### User Story 3 - Comprehensive Processing Audit Log (Priority: P2)

As a data engineer, I need a complete summary log at the end of a batch run that lists every processed exam, whether it was successfully converted, and if not, the specific reason it failed, so I can audit the ingestion process and follow up on missing data.

**Why this priority**: While basic error logging (P1) prevents silent failures, a comprehensive audit log is essential for data provenance and quality assurance at scale.

**Independent Test**: Can be tested by providing a known mix of valid and corrupted series and verifying the summary log accounts for 100% of the input folders with correct statuses.

**Acceptance Scenarios**:

1. **Given** a batch of DICOM series, **When** the processing finishes, **Then** a final audit log is generated containing an entry for every series.
2. **Given** the audit log, **When** a series was successfully converted, **Then** its entry indicates "success".
3. **Given** the audit log, **When** a series failed to convert, **Then** its entry indicates "failed" and includes the reason (e.g., "missing pixel spacing", "corrupted slice").

## Existing Pipeline Context (`data_cleaner.py`)

The current data cleaning and ingestion pipeline (`data_engine/data_cleaner.py`) follows these sequential steps:
1.  **Organize Data**: Scans the raw data path and organizes folder paths and names.
2.  **Validate DICOMs**: Validates the DICOM series within the organized folders.
3.  **Join Classification Data**: Merges validation results with external classification labels from a CSV.
4.  **Check Missing Classes**: Validates that 'OK' exams have a corresponding classification.
5.  **Filter for Conversion**: Filters the validated and classified data to determine which exams are eligible for NIfTI conversion.
6.  **Convert to NIfTI**: Processes and converts the eligible exams from DICOM to NIfTI.
7.  **Generate Output CSV**: Writes a comprehensive summary CSV containing metadata, validation status, and classification data.

### Data Organization Specification (`file_utils.py`)

The data organization step is handled by `file_utils.py` and involves the following specific features:
- **Standardized Naming**: Parses original folder names (expecting a 'bp' prefix followed by digits). If the numeric portion contains non-numeric characters (e.g., 'bp001_v2'), the system MUST extract the first contiguous sequence of digits. The extracted numeric part MUST be converted to an integer and then zero-padded to exactly 3 digits to form the `BPXXX` format (e.g., `bp0001` or `bp1` both become `BP001`). Handles duplicate naming by appending alphabetical suffixes (e.g., `BP001A`, `BP001B`).
- **Missing Case Detection**: Audits the standardized names against an expected range (1-999) and flags missing case numbers.
- **Folder Statistics Gathering**: Analyzes each folder to count direct `.dcm` files, non-empty subfolders, and total items within subfolders.
- **Data Code Assignment**: Categorizes each folder based on its contents into specific operational codes:
  - `READY`: Contains direct `.dcm` items and no subfolders.
  - `SUBFOLDER_PATH`: Contains no direct items but exactly one non-empty subfolder.
  - `EMPTY`: Contains no items and no subfolders.
  - `DUPLICATE_DATA`: Contains multiple non-empty subfolders or a mix of direct items and a non-empty subfolder.
  - `MISSING`: Case number was not found in the raw data.
- **Path Resolution**: Determines the correct relative data path based on the assigned data code to facilitate downstream processing.
- **Data Orchestration**: Merges name mapping, folder statistics, missing cases, data codes, and paths into a comprehensive unified data structure.

### DICOM Validation Specification (`dicom_utils.py`)

The DICOM validation step is handled by `dicom_utils.py` and involves the following specific features:
- **DICOM Metadata Loading**: Loads and reads DICOM headers from valid `.dcm` files while gracefully skipping invalid or corrupted files.
- **Orientation Determination**: Infers the primary anatomical orientation (AXIAL, CORONAL, SAGITTAL, OBLIQUE) by analyzing the `ImageOrientationPatient` vector.
- **Series Integrity Validation**:
  - **Mixed Series Check**: Verifies that all slices in a folder belong to a single `SeriesInstanceUID`.
  - **Scout/Localizer Filtering**: Filters out 'DERIVED' images (like scout scans) to isolate the 'ORIGINAL' main series slices.
  - **Projection-Based Sorting**: Sorts slices spatially along the normal vector computed from the image orientation, ensuring accurate 3D volume reconstruction even if instance numbers are unreliable.
  - **Spacing Analysis**: Calculates inter-slice distances to determine the effective slice thickness. If there is no single, uniform spacing (i.e., variable spacing or gapped sequences), the system MUST fail the series.
  - **Metadata Extraction**: Extracts and records key information including modality, effective slice thickness, patient sex (falling back to 'UNKNOWN' if missing), image dimensions, and calculates the total physical `exam_size`.
  - **Validation Status Codes**: Assigns explicit statuses (e.g., 'OK', 'NO_DCM_FILES', 'MIXED_SERIES_ERROR', 'NO_ORIGINAL_IMAGES_FOUND', 'DUPLICATE_SLICES', 'GAPPED_SEQUENCE', 'VARIABLE_SPACING') based on the integrity checks.
- **Outlier Detection**: Performs statistical outlier detection using the Interquartile Range (IQR) method on the computed `exam_size` of validated 'OK' series, flagging anomalies as 'BELOW_LIMIT' or 'ABOVE_LIMIT'.

### Edge Cases

- What happens when a DICOM series is missing mandatory spatial tags (e.g., Pixel Spacing)? *(Assumption: The tool will log an error and skip the series, similar to a corrupted file).*
- **Mixed Series**: A folder containing multiple distinct DICOM series intermixed (different `SeriesInstanceUID`s) within the same immediate directory MUST fail the entire folder. The system will log a 'MIXED_SERIES_ERROR' and skip to the next folder.
- **Scout Images**: Folders containing scout/localizer images intermixed with original data MUST be filtered to exclude non-diagnostic slices, and the system MUST log a warning indicating that these slices were filtered.
- **Zero Valid Series**: If a batch contains zero valid DICOM series, the system MUST log a WARNING 'NO_VALID_SERIES_FOUND' and complete successfully without errors.
- **1-Slice Series**: If a DICOM series contains only 1 slice (2D image), the system MUST skip the conversion, log a 'NOT_3D_VOLUME_ERROR', and mark the series as failed.
- **Long File Paths**: If the output NIfTI file path exceeds the OS maximum path length, the system MUST handle the `OSError`, log an ERROR, skip the file, and continue.
- **Mixed File Types**: If a folder contains valid DICOMs mixed with non-DICOM files (e.g., `.txt`, `.jpg`), the non-DICOM files MUST be ignored without failing the series.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST process a batch of directories by recursively searching for valid DICOM series and converting each into a single 3D NIfTI-1 (`.nii.gz`) volume.
- **FR-001b**: System MUST process discovered DICOM series sequentially.
- **FR-002**: System MUST parse and preserve `Pixel Spacing` (0028,0030) and `Slice Thickness` (0018,0050) from DICOM headers into the NIfTI affine matrix.
- **FR-003**: System MUST parse and preserve `Image Orientation Patient` (0020,0037) from DICOM headers into the NIfTI affine matrix.
- **FR-004**: System MUST parse and preserve `Image Position Patient` (0020,0032) from DICOM headers into the NIfTI affine matrix.
- **FR-005**: System MUST normalize the output voxel data to a Float32 data type.
- **FR-006**: System MUST catch exceptions caused by corrupted or incomplete DICOM files.
- **FR-007**: System MUST log a clear warning/error message using the Python `logging` module to both standard output and a dedicated log file in JSONL (JSON lines) format (e.g., `ingestion.log`) when a corrupted series is encountered.
- **FR-008**: System MUST skip the creation of a NIfTI file for a corrupted series and proceed sequentially to the next discovered series in the batch without crashing.
- **FR-009**: System MUST output a complete summary log (CSV format) of all processed exams at the end of the batch, explicitly indicating whether each was successfully converted or failed, along with the specific reason for any failure. Numeric fields for physical measurements (e.g., exam size, slice thickness) MUST be standardized to 2 decimal places, and counts MUST be integers. The log MUST be complete and include entries for all cases, including those identified as MISSING during organization.
- **FR-010**: System MUST fail and skip folders containing mixed DICOM series (multiple distinct `SeriesInstanceUID`s) within the same immediate directory, logging a 'MIXED_SERIES_ERROR'.
- **FR-011**: System MUST fail and skip DICOM series missing mandatory spatial tags (Pixel Spacing, Orientation, or Position), logging an ERROR level message 'MISSING_SPATIAL_METADATA_ERROR'.
- **FR-012**: System MUST filter out images where the `ImageType` tag contains 'DERIVED' or 'LOCALIZER', process only 'ORIGINAL' slices, and log a WARNING level message indicating that scout/derived slices were filtered.
- **FR-013**: System MUST name output NIfTI files using the standardized case name (e.g., `BP001.nii.gz`).
- **FR-014**: System MUST handle partial IO failures (e.g., out of disk space) by catching exceptions, logging a CRITICAL error, marking the series as 'FAILED_IO', and attempting to proceed with the next series in the batch.

### Technical Constraints

- **TC-001**: The system MUST NOT modify or delete the source DICOM files. Raw data integrity must be maintained as read-only.
- **TC-002**: All ingestion processes MUST be idempotent. Re-running the tool on the same input directory should yield the same output without creating duplicates or redundant processing, provided the output destination is the same. If a destination file already exists, the conversion is skipped and logged as already processed. This successful skipping MUST be verifiable by checking that the file modification time remains unchanged.
- **TC-003**: Output NIfTI files MUST be written to a destination that does not overwrite or replace the original DICOM files.
- **TC-004**: There are no strict memory constraints or maximum processing time constraints defined for the pipeline.

### Assumptions

- The input data is organized with one distinct DICOM series per directory.
- "Corrupted or incomplete" is defined as files that trigger `pydicom.errors.InvalidDicomError`, files where critical image data arrays (e.g., `PixelData`) are missing or unreadable, or files that cause unhandled exceptions during the MONAI/NiBabel load process.
- The system assumes MONAI version >= 1.5.2 and NiBabel are installed and available.
- It is assumed that if the first valid 'ORIGINAL' slice has required spatial tags, they are consistent throughout the entire series. The system validates the first slice.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid DICOM series in a test batch are converted to NIfTI format with matching physical coordinates (affine matrix validation).
- **SC-002**: Batch processing of 1,000 series completes without crashing when exactly 5 randomly selected series contain artificially corrupted files.
- **SC-003**: 100% of output NIfTI files have a defined voxel data type of Float32.
- **SC-004**: Error logs correctly identify the specific directory or file path of the 5 corrupted series in the test batch.
- **SC-005**: The comprehensive summary log contains exactly 1,000 entries corresponding to the 1,000 input series in the test batch, with 995 marked as success and 5 marked as failed with specific error reasons.
