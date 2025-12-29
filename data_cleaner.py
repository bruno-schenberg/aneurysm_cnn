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

def get_folder_items(path):
    """
    Skips .DS_Store files and returns the count of other files (items)
    within a given directory.
    """
    file_count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    if entry.name == '.DS_Store':
                        try:
                            #os.remove(entry.path)
                            print(f"  - Skipped: {entry.path}")
                        except OSError as e:
                            print(f"  - Error Skipping '{entry.path}': {e}")
                    else:
                        file_count += 1
        return file_count
    except (FileNotFoundError, OSError) as e:
        # Handle cases where the directory doesn't exist or can't be accessed
        print(f"Error: Could not count items in '{path}': {e}")
        return 0

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
    # The fields we will be collecting for each folder.
    stat_fields = ['folder', 'direct_items', 'non_empty_subfolders', 'items_in_subfolders']

    for folder_name in folder_list:
        folder_path = os.path.join(base_path, folder_name)
        items_in_subfolders, non_empty_subfolders = 0, 0
        try:
            direct_files = get_folder_items(folder_path)
            subfolders = get_subfolders(folder_path)

            # Iterate through the found subfolders to get their stats
            for subfolder_name in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                # Count items and sub-subfolders inside the subfolder
                num_items = get_folder_items(subfolder_path)
                num_sub_subfolders = len(get_subfolders(subfolder_path))
                
                # A subfolder is non-empty if it contains files or other folders.
                if num_items > 0 or num_sub_subfolders > 0:
                    non_empty_subfolders += 1
                
                # Accumulate the count of files from within the subfolders.
                items_in_subfolders += num_items

            stats_list.append({
                'folder': folder_name,
                'direct_items': direct_files,
                'non_empty_subfolders': non_empty_subfolders,
                'items_in_subfolders': items_in_subfolders
            })
        except OSError as e:
            print(f"  - Error scanning folder '{folder_path}': {e}")
    return stats_list

if __name__ == "__main__":
    case_folders = get_subfolders(RAW_DATA_PATH)
    print(f"Found {len(case_folders)} subfolders in '{RAW_DATA_PATH}'.")
    
    # 1. Generate the mapping from original to new names.
    name_mapping = generate_new_names(case_folders)
    
    # 2. Get statistics for the original folders.
    folder_stats = get_folder_stats(RAW_DATA_PATH, case_folders)
    
    # 3. Merge statistics into the name mapping data.
    # Create a lookup dictionary for fast access to stats.
    stats_map = {stat['folder']: stat for stat in folder_stats}
    
    for item in name_mapping:
        original_name = item['original_name']
        if original_name in stats_map:
            # Update the dictionary with stats, removing the redundant 'folder' key.
            item.update(stats_map[original_name])
            del item['folder']
            
    # 4. Write the combined data to the CSV.
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = ['original_name', 'fixed_name', 'direct_items', 'non_empty_subfolders', 'items_in_subfolders']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(name_mapping)

    print(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")

    add_missing_cases_to_csv(OUTPUT_CSV_PATH, name_mapping)
