import logging
import os
import re
from collections import defaultdict

logger = logging.getLogger("dicom_ingestion")


def get_subfolders(path: str) -> list[str]:
    """Returns subfolder names within a directory."""
    try:
        with os.scandir(path) as entries:
            subfolders = [entry.name for entry in entries if entry.is_dir()]
        return subfolders
    except FileNotFoundError:
        logger.error(f"Error: Directory not found at '{path}'")
        return []


def count_dcm_files(path: str) -> int:
    """Counts .dcm files directly inside a directory, excluding subdirectories."""
    dcm_file_count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.dcm'):
                    dcm_file_count += 1
        return dcm_file_count
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error: Could not count items in '{path}': {e}")
        return 0


def generate_new_names(folder_list: list[str]) -> list[dict]:
    """
    Generates standardized folder names from a list of original names.
    - Looks for a 'bp' or 'BP' prefix followed by digits.
    - Standardizes the prefix to 'BP'.
    - Pads the numeric part to 3 digits (e.g., 1 -> 001, 20 -> 020).
    - Trims any characters after the numeric part.
    Returns a list of dictionaries with 'original_name' and 'fixed_name'.
    """
    # Sort before assigning suffixes so that duplicates always receive the same
    # suffix regardless of filesystem ordering (e.g., bp001 always → BP001A).
    sorted_folders = sorted(folder_list)

    potential_names = []
    pattern = re.compile(r"^(bp)_?(\d+)", re.IGNORECASE)

    for original_name in sorted_folders:
        match = pattern.match(original_name)
        if match:
            # Convert to int to strip leading zeros, then re-pad to 3 digits.
            number_part = str(int(match.group(2)))
            fixed_name = f"BP{number_part.zfill(3)}"
        else:
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


def find_missing_cases(name_mapping: list[dict]) -> list[dict]:
    """
    Finds missing case numbers (1-999) and returns them as a list of dicts.

    Args:
        name_mapping: The list of dicts already written to the CSV.

    Returns:
        A list of dictionaries for the missing cases.
    """
    logger.info("Checking for missing case numbers...")
    pattern = re.compile(r"BP(\d+)")
    existing_numbers = set()
    for item in name_mapping:
        match = pattern.match(item['fixed_name'])
        if match:
            existing_numbers.add(int(match.group(1)))

    missing_rows = []
    for i in range(1, 1000):
        if i not in existing_numbers:
            fixed_name = f"BP{i:03d}"
            missing_rows.append({'original_name': 'missing', 'fixed_name': fixed_name})
    return missing_rows


def get_folder_stats(base_path: str, folder_list: list[str]) -> list[dict]:
    """
    Counts items and subfolders within a list of specified folders.

    For each folder, it counts:
    1. The number of .dcm files directly inside it.
    2. The number of non-empty subfolders directly inside it.
    3. The total number of .dcm files within all of those subfolders.

    Returns a list of dictionaries, each containing the stats for a folder.
    """
    logger.info("\nGathering folder statistics...")
    stats_list = []

    for folder_name in folder_list:
        folder_path = os.path.join(base_path, folder_name)
        items_in_subfolders, non_empty_subfolders = 0, 0
        non_empty_subfolder_names = []

        try:
            direct_files = count_dcm_files(folder_path)
            subfolders = get_subfolders(folder_path)

            for subfolder_name in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                num_items = count_dcm_files(subfolder_path)
                num_sub_subfolders = len(get_subfolders(subfolder_path))

                if num_items > 0 or num_sub_subfolders > 0:
                    non_empty_subfolder_names.append(subfolder_name)

                items_in_subfolders += num_items

            stats_list.append({
                'folder': folder_name,
                'direct_items': direct_files,
                'non_empty_subfolders': non_empty_subfolder_names,
                'items_in_subfolders': items_in_subfolders
            })
        except OSError as e:
            logger.info(f"  - Error scanning folder '{folder_path}': {e}")
    return stats_list


def add_data_codes(name_mapping: list[dict]) -> list[dict]:
    """
    Adds a 'data_code' to each item in the mapping based on folder stats.

    The logic for the data code is as follows:
    - EMPTY: 0 non-empty subfolders and 0 direct items.
    - READY: 0 non-empty subfolders and >0 direct items.
    - SUBFOLDER_PATH: 1 non-empty subfolder and 0 direct items.
    - DUPLICATE_DATA: 1 non-empty subfolder and >0 direct items.
    - DUPLICATE_DATA: >=2 non-empty subfolders.
    - MISSING: The case was not found in the raw data.

    Args:
        name_mapping: The list of dictionaries, some containing folder stats.

    Returns:
        The updated list of dictionaries with a 'data_code' key.
    """
    logger.info("Assigning data codes...")
    for item in name_mapping:
        if 'direct_items' not in item:
            item['data_code'] = 'MISSING'
            continue

        direct_items = item['direct_items']
        non_empty_subfolders = len(item.get('non_empty_subfolders', []))
        action_code = ''

        if non_empty_subfolders == 0:
            if direct_items == 0:
                action_code = 'EMPTY'
            else:
                action_code = 'READY'
        elif non_empty_subfolders == 1:
            if direct_items == 0:
                action_code = 'SUBFOLDER_PATH'
            else:
                action_code = 'DUPLICATE_DATA'
        else:
            action_code = 'DUPLICATE_DATA'

        item['data_code'] = action_code
    logger.info("Data codes assigned.")
    return name_mapping


def add_data_paths(name_mapping: list[dict]) -> list[dict]:
    """
    Adds a 'data_path' to each item based on its action code.

    The path is relative to the main raw data directory.
    - READY: The original folder name (e.g., 'bp1').
    - SUBFOLDER_PATH: Path to the single non-empty subfolder (e.g., 'bp2/data').
    - Other codes (MISSING, EMPTY, DUPLICATE_DATA): Path is set to the code string.
    """
    logger.info("Determining data paths...")
    for item in name_mapping:
        code = item.get('data_code')
        path = ''

        if code == 'READY':
            path = item['original_name']
        elif code == 'SUBFOLDER_PATH':
            subfolder_name = item['non_empty_subfolders'][0]
            path = os.path.join(item['original_name'], subfolder_name)
        elif code in ('MISSING', 'EMPTY', 'DUPLICATE_DATA'):
            path = code

        item['data_path'] = path

        if 'non_empty_subfolders' in item:
            del item['non_empty_subfolders']

    return name_mapping


def organize_data(case_folders: list[str], RAW_DATA_PATH: str) -> list[dict]:

    # 1. Generate the mapping from original to new names.
    name_mapping = generate_new_names(case_folders)

    # 2. Get statistics for the original folders.
    folder_stats = get_folder_stats(RAW_DATA_PATH, case_folders)

    # 3. Merge statistics into the name mapping data.
    stats_map = {stat['folder']: stat for stat in folder_stats}

    for item in name_mapping:
        original_name = item['original_name']
        if original_name in stats_map:
            item.update(stats_map[original_name])
            del item['folder']

    # 4. Add total_dcms count.
    for item in name_mapping:
        if 'direct_items' in item:
            item['total_dcms'] = item.get('direct_items', 0) + item.get('items_in_subfolders', 0)

    # 5. Find missing cases and add them to our main list.
    missing_cases = find_missing_cases(name_mapping)
    combined_mapping = name_mapping + missing_cases
    logger.info(f"Found {len(missing_cases)} missing case entries.")

    # 6. Add action codes to the complete list (including missing cases).
    data_with_codes = add_data_codes(combined_mapping)

    # 7. Add the final data paths.
    organized_data = add_data_paths(data_with_codes)

    return organized_data
