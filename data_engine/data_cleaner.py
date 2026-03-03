"""
DICOM Ingestion & NIfTI Standardization Pipeline.

This script serves as the primary entry point for processing raw DICOM files
into standardized NIfTI volumes. It performs the following sequential steps:
1. Discovers and organizes DICOM case folders.
2. Validates DICOM series for integrity and completeness.
3. Analyzes and logs mixed series (multiple orientations/modalities).
4. Associates cases with classification data (e.g., from classes.csv).
5. Filters out cases with missing data or failed validations.
6. Converts eligible DICOM series to NIfTI format.
7. Generates comprehensive audit logs and legacy rename mappings.

Usage:
    python data_cleaner.py --raw-dir /path/to/raw --nifti-dir /path/to/output
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

from src.file_utils import get_subfolders, organize_data
from src.dicom_utils import validate_dcms, analyze_mixed_folders
from src.class_utils import join_class_data, check_missing_class
from src.nifti_utils import filter_for_conversion, process_and_convert_exams
from src.logging_utils import setup_logger, write_audit_log


# Base directory configurations
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DATA_PATH = Path("/mnt/data/cases-3/raw")
DEFAULT_OUTPUT_NIFTI_PATH = Path("/mnt/data/cases-3/nifti")

# Output file paths
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = BASE_DIR / "dataset"

OUTPUT_CSV_PATH = OUTPUT_DIR / "folder_rename_map.csv"
CLASSES_CSV_PATH = DATASET_DIR / "classes.csv"
MIXED_SERIES_CSV_PATH = OUTPUT_DIR / "mixed_series_analysis.csv"
INGESTION_LOG_PATH = OUTPUT_DIR / "ingestion.log"
AUDIT_LOG_PATH = OUTPUT_DIR / "ingestion_summary.csv"


def _build_audit_log(conversion_results: List[Dict[str, Any]], final_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Constructs a complete audit log combining successful conversions and skipped/failed items.
    
    Args:
        conversion_results: List of dictionaries detailing successful NIfTI conversions.
        final_data: The full dataset containing all processed and validated cases.
        
    Returns:
        A combined list of audit log entries for every discovered case.
    """
    full_audit_results = []
    
    # First, add the conversion results
    full_audit_results.extend(conversion_results)
    
    # Then, add those that weren't eligible (failed validation or missing class)
    converted_names = {r['exam_name'] for r in conversion_results}
    for item in final_data:
        if item['fixed_name'] not in converted_names:
            full_audit_results.append({
                "exam_name": item['fixed_name'],
                "status": "failed",
                "reason": item.get('validation_status', "Unknown validation error"),
                "output_path": ""
            })
            
    return full_audit_results


def run_pipeline(raw_dir: str | Path, nifti_dir: str | Path) -> None:
    """
    Executes the full DICOM to NIfTI ingestion and standardization pipeline.

    Args:
        raw_dir: Path to the directory containing raw DICOM case folders.
        nifti_dir: Path to the directory where NIfTI outputs should be saved.
    """
    raw_dir_str = str(raw_dir)
    nifti_dir_str = str(nifti_dir)

    # 0. Setup logger
    # Ensure output directory exists before logging
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(str(INGESTION_LOG_PATH))
    logger.info("Starting DICOM Ingestion & NIfTI Standardization Pipeline")

    try:
        case_folders = get_subfolders(raw_dir_str)
        logger.info(f"Found {len(case_folders)} subfolders in '{raw_dir_str}'.")
        
        # 1. Organize the data paths and names
        organized_data = organize_data(case_folders, raw_dir_str)

        # 2. Validate the DICOM series.
        validated_data = validate_dcms(organized_data, raw_dir_str)

        # 2.5 Analyze mixed folders
        analyze_mixed_folders(validated_data, raw_dir_str, str(MIXED_SERIES_CSV_PATH))

        # 3. Join with classification data from classes.csv
        data_with_classes = join_class_data(validated_data, str(CLASSES_CSV_PATH))

        # 4. Check for missing classes in 'OK' exams
        final_data = check_missing_class(data_with_classes)

        # 5. Filter the data for NIfTI conversion before saving
        eligible_for_conversion = filter_for_conversion(final_data)

        # 6. Convert to NIfTI and track results
        conversion_results = process_and_convert_exams(
            eligible_for_conversion, raw_dir_str, nifti_dir_str
        )

        # 7. Write the final audit log (User Story 3)
        full_audit_results = _build_audit_log(conversion_results, final_data)
        write_audit_log(full_audit_results, str(AUDIT_LOG_PATH))
        logger.info(f"Successfully created audit log at '{AUDIT_LOG_PATH}'")

        # 8. Write the folder rename map (legacy support)
        with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
            fieldnames = [
                'original_name', 'fixed_name', 'data_path', 'total_dcms',
                'validation_status', 'duplicate_slice_count', 'scout_slice_count', 'orientation',
                'modality', 'slice_thickness', 'patient_age', 'patient_sex', 'image_dimensions',
                'class', 'location', 'exam_size'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(final_data)

        logger.info(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")
        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM Ingestion & NIfTI Standardization Pipeline")
    parser.add_argument(
        "--raw-dir", 
        default=str(DEFAULT_RAW_DATA_PATH), 
        help=f"Path to raw DICOM data directory (default: {DEFAULT_RAW_DATA_PATH})"
    )
    parser.add_argument(
        "--nifti-dir", 
        default=str(DEFAULT_OUTPUT_NIFTI_PATH), 
        help=f"Path to output NIfTI directory (default: {DEFAULT_OUTPUT_NIFTI_PATH})"
    )
    args = parser.parse_args()

    run_pipeline(Path(args.raw_dir), Path(args.nifti_dir))
