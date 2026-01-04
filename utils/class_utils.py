import csv
import re

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

def check_missing_class(data):
    """
    Checks 'OK' exams and flags those with a missing classification.

    If an exam has a validation_status of 'OK' but no 'class' was joined
    from the classes CSV, its status is updated to 'MISSING_CLASS'.

    Args:
        data (list): The list of dictionaries after class data has been joined.

    Returns:
        list: The updated data list.
    """
    print("\nChecking for missing classifications in 'OK' exams...")
    missing_class_count = 0
    for item in data:
        if item.get('validation_status') == 'OK' and not item.get('class'):
            item['validation_status'] = 'MISSING_CLASS'
            missing_class_count += 1

    print(f"  - Flagged {missing_class_count} 'OK' exams with status 'MISSING_CLASS'.")
    return data