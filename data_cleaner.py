import os
import re
import csv
from collections import defaultdict
import pydicom
import numpy as np
from collections import Counter

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
    Counts the number of .dcm files within a given directory.
    """
    dcm_file_count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                # Check if the entry is a file and ends with .dcm (case-insensitive)
                if entry.is_file() and entry.name.lower().endswith('.dcm'):
                    dcm_file_count += 1
        return dcm_file_count
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

def find_missing_cases(name_mapping):
    """
    Finds missing case numbers (1-999) and returns them as a list of dicts.

    Args:
        name_mapping (list): The list of dicts already written to the CSV.

    Returns:
        list: A list of dictionaries for the missing cases.
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
    for i in range(1, 1000):  # Check for numbers 1 through 999
        if i not in existing_numbers:
            fixed_name = f"BP{i:03d}"  # e.g., 24 -> BP024
            missing_rows.append({'original_name': 'missing', 'fixed_name': fixed_name})
    return missing_rows

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
        non_empty_subfolder_names = []

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
                    non_empty_subfolder_names.append(subfolder_name)
                
                # Accumulate the count of files from within the subfolders.
                items_in_subfolders += num_items

            stats_list.append({
                'folder': folder_name,
                'direct_items': direct_files,
                'non_empty_subfolders': non_empty_subfolder_names,
                'items_in_subfolders': items_in_subfolders
            })
        except OSError as e:
            print(f"  - Error scanning folder '{folder_path}': {e}")
    return stats_list

def add_data_codes(name_mapping):
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
        name_mapping (list): The list of dictionaries, some containing folder stats.

    Returns:
        list: The updated list of dictionaries with a 'data_code' key.
    """
    print("Assigning data codes...")
    for item in name_mapping:
        # Skip 'missing' entries that were added later and have no stats
        if 'direct_items' not in item:
            item['data_code'] = 'MISSING'
            continue

        direct_items = item['direct_items']
        non_empty_subfolders = len(item.get('non_empty_subfolders', []))
        action_code = ''

        if non_empty_subfolders == 0:
            if direct_items == 0:
                action_code = 'EMPTY'
            else:  # direct_items > 0
                action_code = 'READY'
        elif non_empty_subfolders == 1:
            if direct_items == 0:
                action_code = 'SUBFOLDER_PATH'
            else:  # direct_items > 0
                action_code = 'DUPLICATE_DATA'
        else:  # non_empty_subfolders >= 2
            action_code = 'DUPLICATE_DATA'

        item['data_code'] = action_code
    print("Data codes assigned.")
    return name_mapping

def add_data_paths(name_mapping):
    """
    Adds a 'data_path' to each item based on its action code.

    The path is relative to the main raw data directory.
    - READY: The original folder name (e.g., 'bp1').
    - SUBFOLDER_PATH: Path to the single non-empty subfolder (e.g., 'bp2/data').
    - Other codes (MISSING, EMPTY, DUPLICATE_DATA): Path is empty.
    """
    print("Determining data paths...")
    for item in name_mapping:
        code = item.get('data_code')
        path = ''  # Default to empty path

        if code == 'READY':
            path = item['original_name']
        elif code == 'SUBFOLDER_PATH':
            # Path is the single non-empty subfolder.
            # The check for len == 1 is implicitly handled by the data code logic.
            subfolder_name = item['non_empty_subfolders'][0]
            path = os.path.join(item['original_name'], subfolder_name)
        elif code in ('MISSING', 'EMPTY', 'DUPLICATE_DATA'):
            path = code
        
        item['data_path'] = path

        # Clean up the subfolder list from the final output if it exists
        if 'non_empty_subfolders' in item:
            del item['non_empty_subfolders']

    return name_mapping

def get_orientation(iop):
    """Determines if the orientation is Axial, Coronal, Sagittal, or Oblique."""
    if iop is None or len(iop) < 6:
        return "UNKNOWN"
    
    # Rounding to nearest integer to handle slight scanner tilts
    r = [round(x) for x in iop]
    
    # Standard DICOM directions: [x1, y1, z1, x2, y2, z2]
    if r == [1, 0, 0, 0, 1, 0]:
        return "AXIAL"
    elif r == [1, 0, 0, 0, 0, -1]:
        return "CORONAL"
    elif r == [0, 1, 0, 0, 0, -1]:
        return "SAGITTAL"
    else:
        return "OBLIQUE"

def validate_dcms(data_mapping, base_path):
    print("\nValidating DICOM series...")
    
    for item in data_mapping:
        data_path = item.get('data_path')
        if data_path in ('EMPTY', 'MISSING', 'DUPLICATE_DATA'):
            item['validation_status'] = 'NOT_APPLICABLE'
            continue

        print(f"  - Validating {item['fixed_name']}...")

        full_path = os.path.join(base_path, data_path)
        try:
            # 1. Load all headers
            metadata_list = []
            for f in os.listdir(full_path):
                if f.lower().endswith('.dcm'):
                    ds = pydicom.dcmread(os.path.join(full_path, f), stop_before_pixels=True)
                    metadata_list.append(ds)

            if not metadata_list:
                item['validation_status'] = 'NO_DCM_FILES'
                continue

            # 2. Check for Series Consistency (SeriesInstanceUID)
            uids = [ds.SeriesInstanceUID for ds in metadata_list]
            unique_uids = set(uids)
            if len(unique_uids) > 1:
                # Count which UID is most common to help the user debug
                occurences = Counter(uids)
                item['validation_status'] = f'MIXED_SERIES_ERROR ({len(unique_uids)} UIDs found)'
                continue

            # 3. Sort by physical Z-axis (ImagePositionPatient)
            # Use the index 2 (Z) for Axial, but for Sagittal you might need index 0 (X)
            # A more robust sort uses the dot product with the normal vector, 
            # but for most clinical work, sorting by Z-coord is standard.
            metadata_list.sort(key=lambda x: float(x.ImagePositionPatient[2]))

            # 4. Spacing Analysis (Geometric)
            z_positions = [float(ds.ImagePositionPatient[2]) for ds in metadata_list]
            z_diffs = np.around(np.abs(np.diff(z_positions)), decimals=3)
            unique_spacings = np.unique(z_diffs)
            
            sample_ds = metadata_list[len(metadata_list)//2]
            declared_spacing = getattr(sample_ds, 'SpacingBetweenSlices', sample_ds.get('SliceThickness', 0))

            item['spacing_type'] = "CONSISTENT" if len(unique_spacings) <= 1 else "VARIABLE"
            item['orientation'] = get_orientation(sample_ds.get('ImageOrientationPatient'))
            
            # 5. Integrity Logic
            instance_numbers = [int(ds.InstanceNumber) for ds in metadata_list]
            is_sequential = all(np.diff(sorted(instance_numbers)) == 1)

            if len(set(instance_numbers)) != len(instance_numbers):
                status = 'DUPLICATE_INSTANCES'
            elif item['spacing_type'] == "CONSISTENT" and len(z_diffs) > 0:
                # Compare physical spacing to header spacing
                if not np.isclose(unique_spacings[0], float(declared_spacing), atol=1e-2):
                    status = 'SPACING_MISMATCH'
                else:
                    status = 'OK'
            elif not is_sequential:
                status = 'GAPPED_SEQUENCE'
            else:
                status = 'OK'

            item['validation_status'] = status

        except Exception as e:
            item['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

    print("Validation complete.")
    return data_mapping

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

    # 4. Add total_dcms count.
    for item in name_mapping:
        # This key will only exist for non-missing cases that had stats.
        if 'direct_items' in item:
            item['total_dcms'] = item.get('direct_items', 0) + item.get('items_in_subfolders', 0)

    # 5. Find missing cases and add them to our main list.
    missing_cases = find_missing_cases(name_mapping)
    combined_mapping = name_mapping + missing_cases
    print(f"Found {len(missing_cases)} missing case entries.")

    # 6. Add action codes to the complete list (including missing cases).
    data_with_codes = add_data_codes(combined_mapping)

    # 7. Add the final data paths.
    final_data = add_data_paths(data_with_codes)

    # 8. Validate the DICOM series.
    validated_data = validate_dcms(final_data, RAW_DATA_PATH)

    # 9. Write the final, combined data to the CSV in one go.
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        # Define the order of columns for the CSV file.
        fieldnames = ['original_name', 'fixed_name', 'data_path', 'total_dcms',
                      'validation_status', 'series_description',
                      'modality', 'slice_thickness', 'is_axial']
        # Use `extrasaction='ignore'` to prevent errors for rows
        # that don't have all the stat fields.
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_data)

    print(f"Successfully created rename map at '{OUTPUT_CSV_PATH}'")
