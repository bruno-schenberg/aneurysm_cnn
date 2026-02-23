# Quickstart: Testing DICOM to NIfTI Pipeline

This quickstart guides you on how to test the newly refactored DICOM to NIfTI pipeline.

## Prerequisites
Ensure the environment is set up and MONAI is installed.
```bash
pip install -r data_engine/requirements.txt
```
*(Note: `monai` must be added to the requirements.txt)*

## Running the Ingestion Pipeline

To execute the entire data cleaning and NIfTI conversion process:

```bash
cd data_engine
python data_cleaner.py
```

### What to Expect
1. **Console Output**: You will see a stream of standard python `logging` output detailing which exams are being discovered, processed, skipped (due to idempotency), or failed (due to corruption, mixed series, or missing spatial tags). Warnings for scout image filtering will also be visible.
2. **Log File**: A detailed technical log will be generated at `data_engine/ingestion.log`. This file captures all `WARNING` and `ERROR` stack traces from MONAI without polluting the console.
3. **Summary CSV**: Upon completion, a comprehensive audit log will be generated at `data_engine/ingestion_summary.csv`. You can open this file to see a complete tally of all 1,000+ exams and their precise conversion statuses.
4. **Output Data**: NIfTI (`.nii.gz`) files will be safely written to the specified output directory (e.g., `/mnt/data/cases-3/nifti`), named using the standardized case naming convention (e.g., `BP001.nii.gz`), without ever altering the original raw DICOM files.