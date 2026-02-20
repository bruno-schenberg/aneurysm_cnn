# Feature Specification: DICOM Ingestion & NIfTI Standardization

**Feature Branch**: `001-dicom-to-nifti`  
**Created**: 2026-02-20  
**Status**: Draft  
**Input**: User description: "create a new feature branch number 1. ## 1. Feature: DICOM Ingestion & NIfTI Standardization **Goal**: Robustly convert raw DICOM series into standard NIfTI-1 files, handling real-world data issues and ensuring header integrity. ### Requirements * **[REQ-1.1]** Convert raw DICOM series (multiple slices) into a single 3D NIfTI (`.nii.gz`) volume. * **[REQ-1.2]** Corrupted or incomplete DICOM files MUST be logged as warnings and skipped. The process MUST be resilient and not crash the entire batch. * **[REQ-1.3]** Reliably parse and preserve spatial metadata from DICOM tags: * **Pixel Spacing**: (0028,0030) and Slice Thickness (0018,0050). * **Orientation**: Image Orientation Patient (0020,0037). * **Position**: Image Position Patient (0020,0032). * **[REQ-1.4]** Output volumes MUST be normalized to Float32 data type. ### Acceptance Criteria * **[AC-1.1]** Given a valid DICOM series folder, the tool produces a NIfTI file with an affine matrix that matches the physical coordinates of the source. * **[AC-1.2]** When a series contains a corrupted slice, a clear error is logged, the output file is not created (or marked as invalid), and the next series is processed."

## Clarifications

### Session 2026-02-20
- Q: How should batch processing handle multiple series? → A: Sequentially (simpler, lower memory footprint)
- Q: How should the tool discover DICOM series in a batch? → A: Recursively search all subdirectories for valid DICOM series
- Q: How should error logging be handled? → A: Python `logging` module to stdout and a file (e.g., `ingestion.log`)

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

**Independent Test**: Can be tested by running a batch with a known mix of valid and corrupted series and verifying the summary log accounts for 100% of the input folders with correct statuses.

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

### Edge Cases

- What happens when a DICOM series is missing mandatory spatial tags (e.g., Pixel Spacing)? *(Assumption: The tool will log an error and skip the series, similar to a corrupted file).*
- How does the system handle a folder containing multiple distinct DICOM series intermixed? *(Assumption: The tool expects one series per folder. Mixing series may result in an error or undefined behavior, which should be logged).*

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST process a batch of directories by recursively searching for valid DICOM series and converting each into a single 3D NIfTI-1 (`.nii.gz`) volume.
- **FR-001b**: System MUST process discovered DICOM series sequentially.
- **FR-002**: System MUST parse and preserve `Pixel Spacing` (0028,0030) and `Slice Thickness` (0018,0050) from DICOM headers into the NIfTI affine matrix.
- **FR-003**: System MUST parse and preserve `Image Orientation Patient` (0020,0037) from DICOM headers into the NIfTI affine matrix.
- **FR-004**: System MUST parse and preserve `Image Position Patient` (0020,0032) from DICOM headers into the NIfTI affine matrix.
- **FR-005**: System MUST normalize the output voxel data to a Float32 data type.
- **FR-006**: System MUST catch exceptions caused by corrupted or incomplete DICOM files.
- **FR-007**: System MUST log a clear warning/error message using the Python `logging` module to both standard output and a dedicated log file (e.g., `ingestion.log`) when a corrupted series is encountered.
- **FR-008**: System MUST skip the creation of a NIfTI file for a corrupted series and proceed sequentially to the next discovered series in the batch without crashing.
- **FR-009**: System MUST output a complete summary log (CSV format) of all processed exams at the end of the batch, explicitly indicating whether each was successfully converted or failed, along with the specific reason for any failure.

### Technical Constraints

- **TC-001**: The system MUST NOT modify or delete the source DICOM files. Raw data integrity must be maintained as read-only.
- **TC-002**: All ingestion processes MUST be idempotent. Re-running the tool on the same input directory should yield the same output without creating duplicates or redundant processing, provided the output destination is the same.
- **TC-003**: Output NIfTI files MUST be written to a destination that does not overwrite or replace the original DICOM files.

### Assumptions

- The input data is organized with one distinct DICOM series per directory.
- "Corrupted or incomplete" includes files that cannot be read by standard DICOM parsing libraries or are missing critical image data arrays.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid DICOM series in a test batch are converted to NIfTI format with matching physical coordinates (affine matrix validation).
- **SC-002**: Batch processing of 1,000 series completes without crashing when exactly 5 randomly selected series contain artificially corrupted files.
- **SC-003**: 100% of output NIfTI files have a defined voxel data type of Float32.
- **SC-004**: Error logs correctly identify the specific directory or file path of the 5 corrupted series in the test batch.
- **SC-005**: The comprehensive summary log contains exactly 1,000 entries corresponding to the 1,000 input series in the test batch, with 995 marked as success and 5 marked as failed with specific error reasons.