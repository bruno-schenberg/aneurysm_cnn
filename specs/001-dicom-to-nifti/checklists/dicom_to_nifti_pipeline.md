# Checklist: dicom_to_nifti_pipeline.md

**Purpose**: Unit Tests for Requirements - Validating the quality, clarity, and completeness of the DICOM to NIfTI pipeline specifications.
**Created**: 2026-02-21

## Requirement Completeness
- [x] CHK001 - Are the conditions for calculating the effective slice thickness explicitly defined (e.g., fallback if multiple spacings exist)? [Gap, Spec §Data Organization]
- [x] CHK002 - Is the handling of patient sex metadata defined if it is missing from the DICOM header? [Gap, Spec §DICOM Validation]
- [x] CHK003 - Are the data type requirements for all numeric fields in the CSV audit log specified? [Gap, Spec §FR-009]

## Requirement Clarity
- [x] CHK004 - Is the term "corrupted or incomplete" quantified or defined with specific failure modes (e.g., EOF error, missing critical arrays)? [Clarity, Spec §Assumptions]
- [x] CHK005 - Are the rounding rules explicitly defined for creating the `BPXXX` format from original folder names? [Clarity, Spec §Data Organization]
- [x] CHK006 - Is the format of the generated `ingestion.log` (e.g., JSON, plain text) explicitly specified? [Clarity, Spec §Clarifications]

## Requirement Consistency
- [x] CHK007 - Is the treatment of "missing cases" consistent between the data code assignment and the final audit log output? [Consistency, Spec §Data Organization & FR-009]
- [x] CHK008 - Do the error logging requirements consistently state the exact log level (e.g., WARNING vs ERROR) for missing spatial metadata vs scout image filtering? [Consistency, Spec §FR-011 & FR-012]

## Acceptance Criteria Quality
- [x] CHK009 - Is "matches the physical coordinates of the source" [AC-1.1] objectively measurable (e.g., to a specific decimal precision)? [Measurability]
- [x] CHK010 - Can the successful skipping of an idempotent run [TC-002] be objectively verified via test automation? [Measurability]

## Scenario Coverage
- [x] CHK011 - Are requirements defined for how the system handles a batch containing zero valid DICOM series? [Coverage, Edge Case]
- [x] CHK012 - Are the requirements for partial failure defined (e.g., out of disk space halfway through a batch)? [Coverage, Exception Flow]
- [x] CHK013 - Are requirements specified for handling non-numeric characters in the numeric portion of the original 'bp' folder names? [Coverage, Edge Case]

## Edge Case Coverage
- [x] CHK014 - Is the required behavior specified if a DICOM series contains only 1 slice (2D image)? [Edge Case, Gap]
- [x] CHK015 - Is it explicitly documented what happens when the output NIfTI file path exceeds the maximum path length of the OS? [Edge Case, Gap]
- [x] CHK016 - Is the expected behavior defined if a folder contains valid DICOMs mixed with non-DICOM files (e.g., `.txt`, `.jpg`)? [Edge Case, Gap]

## Non-Functional Requirements
- [x] CHK017 - Are memory constraint requirements defined for loading large series (e.g., 2000+ slices) using MONAI? [Gap, Performance]
- [x] CHK018 - Are maximum processing time requirements defined for a typical batch of 1,000 cases? [Gap, Performance]

## Dependencies & Assumptions
- [x] CHK019 - Are the specific versions or minimum versions of MONAI and NiBabel required by the system documented? [Assumption]
- [x] CHK020 - Is the assumption that all required spatial tags are present in the 'ORIGINAL' slices validated against the data source? [Assumption, Spec §DICOM Validation]

## Ambiguities & Conflicts
- [x] CHK021 - Does the "Fail the entire folder" [FR-010] rule conflict with any requirement to process valid subfolders within that folder? [Conflict]
- [x] CHK022 - Is it ambiguous how the system differentiates between 'DERIVED' images that are scouts versus other types of derived images? [Ambiguity, Spec §FR-012]
