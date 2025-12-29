import csv
from utils.file_utils import get_subfolders, organize_data
from utils.dicom_utils import validate_dcms

RAW_DATA_PATH = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "folder_rename_map.csv"

if __name__ == "__main__":
    case_folders = get_subfolders(RAW_DATA_PATH)
    print(f"Found {len(case_folders)} subfolders in '{RAW_DATA_PATH}'.")
    
    # 1. Organize the data paths and names
    organized_data = organize_data(case_folders, RAW_DATA_PATH)

    # 2. Validate the DICOM series.
    validated_data = validate_dcms(organized_data, RAW_DATA_PATH)

    # 3. Write the final, combined data to the CSV in one go.
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        # Define the order of columns for the CSV file.
        fieldnames = ['original_name', 'fixed_name', 'data_path', 'total_dcms',
                      'validation_status', 'duplicate_slice_count', 'scout_slice_count', 'orientation', 'series_description',
                      'modality', 'slice_thickness']
        # Use `extrasaction='ignore'` to prevent errors for rows
        # that don't have all the stat fields.
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(validated_data)

    print(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")
