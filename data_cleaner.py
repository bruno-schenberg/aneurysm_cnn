import csv
import re
from utils.file_utils import get_subfolders, organize_data
from utils.dicom_utils import validate_dcms

RAW_DATA_PATH = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "folder_rename_map.csv"
CLASSES_CSV_PATH = "classes.csv"

def join_class_data(validated_data, classes_csv_path):
    """
    Joins classification data from classes.csv to the validated data list.

    It matches 'fixed_name' from the validated data with 'exam' from the CSV.
    It handles cases where 'fixed_name' has a suffix (e.g., 'BP123A') by
    matching it to the base name in the CSV (e.g., 'BP123').

    Args:
        validated_data (list): The list of dictionaries containing validated data.
        classes_csv_path (str): The path to the classes.csv file.

    Returns:
        list: The validated_data list, with 'class' and 'location' added.
    """
    print(f"\nJoining data from '{classes_csv_path}'...")
    class_map = {}
    try:
        with open(classes_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Standardize key to uppercase, e.g., 'bp001' -> 'BP001'
                exam_name = row.get('exam', '').strip().upper()
                if exam_name:
                    class_map[exam_name] = {
                        'class': row.get('class'),
                        'location': row.get('location'),
                        'patient_age': row.get('Age')
                    }
    except FileNotFoundError:
        print(f"  - Warning: Classes file not found at '{classes_csv_path}'. Skipping join.")
        return validated_data

    for item in validated_data:
        # Match 'BP123A' or 'BP123' to 'BP123' in the class_map
        match = re.match(r"(BP\d+)", item.get('fixed_name', ''))
        if match:
            base_name = match.group(1)
            class_info = class_map.get(base_name, {})
            item.update(class_info)
    return validated_data

if __name__ == "__main__":
    case_folders = get_subfolders(RAW_DATA_PATH)
    print(f"Found {len(case_folders)} subfolders in '{RAW_DATA_PATH}'.")
    
    # 1. Organize the data paths and names
    organized_data = organize_data(case_folders, RAW_DATA_PATH)

    # 2. Validate the DICOM series.
    validated_data = validate_dcms(organized_data, RAW_DATA_PATH)

    # 3. Join with classification data from classes.csv
    final_data = join_class_data(validated_data, CLASSES_CSV_PATH)

    # 4. Write the final, combined data to the CSV in one go.
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        # Define the order of columns for the CSV file.
        fieldnames = ['original_name', 'fixed_name', 'data_path', 'total_dcms',
                      'validation_status', 'duplicate_slice_count', 'scout_slice_count', 'orientation',
                      'modality', 'slice_thickness', 'patient_age', 'patient_sex', 'image_dimensions',
                      'class', 'location']

        # Use `extrasaction='ignore'` to prevent errors for rows
        # that don't have all the stat fields.
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_data)

    print(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")
