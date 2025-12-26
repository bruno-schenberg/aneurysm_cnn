import os
import re
import csv

RAW_DATA_PATH = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "/home/aneurysm_cnn/folder_rename_map.csv"

def get_subfolders(path):
    """
    Returns a list of subfolder names within a given directory.
    """
    try:
        with os.scandir(path) as entries:
            subfolders = [entry.name for entry in entries if entry.is_dir()]
        return subfolders
    except FileNotFoundError:
        print(f"Error: Directory not found at '{path}'")
        return []

def generate_new_names(folder_list):
    """
    Generates new, standardized folder names from a list of original names.
    - Looks for a 'bp' or 'BP' prefix followed by digits.
    - Standardizes the prefix to 'BP'.
    - Pads the numeric part to 3 digits (e.g., 1 -> 001, 20 -> 020).
    - Trims any characters after the numeric part.
    Returns a list of dictionaries with 'original_name' and 'fixed_name'.
    """
    rename_map = []
    # Regex to find 'bp' (case-insensitive) at the start, followed by digits.
    pattern = re.compile(r"^(bp)(\d+)", re.IGNORECASE)

    for original_name in folder_list:
        match = pattern.match(original_name)
        if match:
            # Extract the numeric part from the match.
            number_part = match.group(2)
            # Format the new name: 'BP' + zero-padded 3-digit number.
            fixed_name = f"BP{number_part.zfill(3)}"
        else:
            # If the pattern doesn't match, keep the original name.
            fixed_name = original_name
        rename_map.append({"original_name": original_name, "fixed_name": fixed_name})
    return rename_map

if __name__ == "__main__":
    case_folders = get_subfolders(RAW_DATA_PATH)
    print(f"Found {len(case_folders)} subfolders in '{RAW_DATA_PATH}'.")

    name_mapping = generate_new_names(case_folders)

    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = ['original_name', 'fixed_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(name_mapping)

    print(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")