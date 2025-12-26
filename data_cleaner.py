import os
import re
import csv
from collections import defaultdict

RAW_DATA_PATH = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "folder_rename_map.csv"

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
    # Sort the folder list to ensure deterministic suffix assignment (A, B, C...)
    # for duplicates, as requested.
    sorted_folders = sorted(folder_list)

    potential_names = []
    # Regex to find 'bp' (case-insensitive) at the start, followed by digits.
    pattern = re.compile(r"^(bp)(\d+)", re.IGNORECASE)

    for original_name in sorted_folders:
        match = pattern.match(original_name)
        if match:
            # Convert to int and back to str to handle leading zeros (e.g., '0100' -> '100').
            number_part = str(int(match.group(2)))
            # Format the new name: 'BP' + zero-padded 3-digit number.
            fixed_name = f"BP{number_part.zfill(3)}"
        else:
            # If the pattern doesn't match, keep the original name.
            fixed_name = original_name
        potential_names.append({"original_name": original_name, "fixed_name": fixed_name})

    # --- Handle duplicate fixed names ---
    fixed_name_counts = defaultdict(int)
    for item in potential_names:
        fixed_name_counts[item['fixed_name']] += 1

    rename_map = []
    suffix_counters = defaultdict(int)
    for item in potential_names:
        base_name = item['fixed_name']
        if fixed_name_counts[base_name] > 1:
            suffix = chr(ord('A') + suffix_counters[base_name])
            item['fixed_name'] = f"{base_name}{suffix}"
            suffix_counters[base_name] += 1
        rename_map.append(item)

    return rename_map

def add_missing_cases_to_csv(csv_path, name_mapping):
    """
    Finds missing case numbers (1-999) and appends them to the CSV.

    Args:
        csv_path (str): The path to the CSV file to append to.
        name_mapping (list): The list of dicts already written to the CSV.
    """
    print("Checking for missing case numbers...")
    # Regex to extract the number from names like 'BP024' or 'BP024A'
    pattern = re.compile(r"BP(\d+)")
    existing_numbers = set()

    for item in name_mapping:
        match = pattern.match(item['fixed_name'])
        if match:
            existing_numbers.add(int(match.group(1)))

    missing_rows = []
    for i in range(1, 1000): # Check for numbers 1 through 999
        if i not in existing_numbers:
            fixed_name = f"BP{i:03d}" # e.g., 24 -> BP024
            missing_rows.append({'original_name': 'missing', 'fixed_name': fixed_name})

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['original_name', 'fixed_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows(missing_rows)

    print(f"Appended {len(missing_rows)} missing case entries to '{csv_path}'")

def get_folder_stats(base_path, folder_list):
    """
    Counts items and subfolders within a list of specified folders.

    For each folder, it counts:
    1. The number of items directly inside it.
    2. The number of subfolders directly inside it.
    3. The total number of items within all of those subfolders.

    Returns a list of dictionaries, each containing the stats for a folder.
    """
    print("\nGathering folder statistics...")
    stats_list = []
    for folder_name in folder_list:
        folder_path = os.path.join(base_path, folder_name)
        direct_files, subfolder_count, items_in_subfolders = 0, 0, 0

        try:
            with os.scandir(folder_path) as entries:
                subfolders_to_scan = []
                for entry in entries:
                    if entry.is_dir():
                        subfolder_count += 1
                        subfolders_to_scan.append(entry.path)
                    elif entry.is_file():
                        direct_files += 1

            for subfolder_path in subfolders_to_scan:
                with os.scandir(subfolder_path) as sub_entries:
                    items_in_subfolders += sum(1 for _ in sub_entries)

            stats_list.append({
                'folder': folder_name,
                'direct_items': direct_files,
                'subfolders': subfolder_count,
                'items_in_subfolders': items_in_subfolders
            })
        except FileNotFoundError:
            print(f"  - Warning: Could not find folder '{folder_path}' to gather stats.")
        except OSError as e:
            print(f"  - Error scanning folder '{folder_path}': {e}")
    return stats_list

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

    add_missing_cases_to_csv(OUTPUT_CSV_PATH, name_mapping)

    # Get and print statistics for each folder
    folder_stats = get_folder_stats(RAW_DATA_PATH, case_folders)
    if folder_stats:
        print("Folder statistics summary:")
        for stats in folder_stats:
            print(f"  - {stats['folder']}:")
            print(f"    - Items directly inside: {stats['direct_items']}")
            print(f"    - Subfolders: {stats['subfolders']}")
            print(f"    - Items in all subfolders: {stats['items_in_subfolders']}")