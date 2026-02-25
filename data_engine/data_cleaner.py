import csv
import os
import argparse
from src.file_utils import get_subfolders, organize_data
from src.dicom_utils import validate_dcms, analyze_mixed_folders
from src.class_utils import join_class_data, check_missing_class
from src.nifti_utils import filter_for_conversion, process_and_convert_exams
from src.logging_utils import setup_logger, write_audit_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DATA_PATH = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "output", "folder_rename_map.csv")
CLASSES_CSV_PATH = os.path.join(BASE_DIR, "dataset", "classes.csv")
MIXED_SERIES_CSV_PATH = os.path.join(BASE_DIR, "output", "mixed_series_analysis.csv")
DEFAULT_OUTPUT_NIFTI_PATH = "/mnt/data/cases-3/nifti"
INGESTION_LOG_PATH = os.path.join(BASE_DIR, "output", "ingestion.log")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "output", "ingestion_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM Ingestion & NIfTI Standardization Pipeline")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DATA_PATH, help="Path to raw DICOM data directory")
    parser.add_argument("--nifti-dir", default=DEFAULT_OUTPUT_NIFTI_PATH, help="Path to output NIfTI directory")
    args = parser.parse_args()

    RAW_DATA_PATH = args.raw_dir
    OUTPUT_NIFTI_PATH = args.nifti_dir

    # 0. Setup logger
    logger = setup_logger(INGESTION_LOG_PATH)
    logger.info("Starting DICOM Ingestion & NIfTI Standardization Pipeline")

    try:
        case_folders = get_subfolders(RAW_DATA_PATH)
        logger.info(f"Found {len(case_folders)} subfolders in '{RAW_DATA_PATH}'.")
        
        # 1. Organize the data paths and names
        organized_data = organize_data(case_folders, RAW_DATA_PATH)

        # 2. Validate the DICOM series.
        validated_data = validate_dcms(organized_data, RAW_DATA_PATH)

        # 2.5 Analyze mixed folders
        analyze_mixed_folders(validated_data, RAW_DATA_PATH, MIXED_SERIES_CSV_PATH)

        # 3. Join with classification data from classes.csv
        data_with_classes = join_class_data(validated_data, CLASSES_CSV_PATH)

        # 4. Check for missing classes in 'OK' exams
        final_data = check_missing_class(data_with_classes)

        # 5. Filter the data for NIfTI conversion before saving
        eligible_for_conversion = filter_for_conversion(final_data)

        # 6. Convert to NIfTI and track results
        conversion_results = process_and_convert_exams(eligible_for_conversion, RAW_DATA_PATH, OUTPUT_NIFTI_PATH)

        # 7. Write the final audit log (User Story 3)
        # We need to include skipped/failed exams that weren't even eligible for conversion
        # but the spec says "100% of discovered folders"
        
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
        
        write_audit_log(full_audit_results, AUDIT_LOG_PATH)
        logger.info(f"Successfully created audit log at '{AUDIT_LOG_PATH}'")

        # 8. Write the folder rename map (legacy support)
        with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
            fieldnames = ['original_name', 'fixed_name', 'data_path', 'total_dcms',
                          'validation_status', 'duplicate_slice_count', 'scout_slice_count', 'orientation',
                          'modality', 'slice_thickness', 'patient_age', 'patient_sex', 'image_dimensions',
                          'class', 'location', 'exam_size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(final_data)

        logger.info(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")
        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
