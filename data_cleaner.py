import csv
from utils.file_utils import get_subfolders, organize_data
from utils.dicom_utils import validate_dcms
from utils.class_utils import join_class_data, check_missing_class
from utils.nifti_utils import filter_for_conversion

RAW_DATA_PATH = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "folder_rename_map.csv"
CLASSES_CSV_PATH = "classes.csv"

if __name__ == "__main__":
    case_folders = get_subfolders(RAW_DATA_PATH)
    print(f"Found {len(case_folders)} subfolders in '{RAW_DATA_PATH}'.")
    
    # 1. Organize the data paths and names
    organized_data = organize_data(case_folders, RAW_DATA_PATH)

    # 2. Validate the DICOM series.
    validated_data = validate_dcms(organized_data, RAW_DATA_PATH)

    # 3. Join with classification data from classes.csv
    data_with_classes = join_class_data(validated_data, CLASSES_CSV_PATH)

    # 4. Check for missing classes in 'OK' exams
    final_data = check_missing_class(data_with_classes)

    # 5. (Example) Filter the data for NIfTI conversion before saving
    # The `eligible_for_conversion` list can now be passed to a conversion function.
    eligible_for_conversion = filter_for_conversion(final_data)

    # 6. Write the final, combined data to the CSV in one go.
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        # Define the order of columns for the CSV file.
        fieldnames = ['original_name', 'fixed_name', 'data_path', 'total_dcms',
                      'validation_status', 'duplicate_slice_count', 'scout_slice_count', 'orientation',
                      'modality', 'slice_thickness', 'patient_age', 'patient_sex', 'image_dimensions',
                      'class', 'location', 'exam_size']

        # Use `extrasaction='ignore'` to prevent errors for rows
        # that don't have all the stat fields.
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_data)

    print(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")
