import os
import csv
import numpy as np
from collections import defaultdict
from utils.dicom_utils import load_dicom_metadata, get_orientation


# --- Configuration ---
INPUT_CSV_PATH = "folder_rename_map.csv"
BASE_DATA_DIR = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "mixed_series_analysis.csv"
# -------------------

def analyze_mixed_series(folder_path):
    """
    Analyzes a directory with mixed DICOM series by grouping files by SeriesInstanceUID
    and running validation on each individual series.

    This is useful for folders flagged with 'MIXED_SERIES_ERROR' to determine
    if any of the contained series are complete and usable.

    Args:
        folder_path (str): The full path to the directory to analyze.

    Returns:
        list: A list of dictionaries, where each dictionary is a summary
              report for a single series found in the folder.
    """
    print(f"\n--- Analyzing Mixed Series in: {folder_path} ---")
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found.")
        return []

    # 1. Load all metadata and group by SeriesInstanceUID
    all_metadata = load_dicom_metadata(folder_path)
    if not all_metadata:
        print("No DICOM files found in this directory.")
        return []

    series_groups = defaultdict(list)
    for ds in all_metadata:
        series_groups[ds.SeriesInstanceUID].append(ds)

    print(f"Found {len(series_groups)} unique series.")
    
    series_reports = []

    # 2. Process each series group individually
    for i, (uid, series_metadata) in enumerate(series_groups.items()):
        report = {
            'series_index': i + 1,
            'series_instance_uid': uid,
            'total_dcms': len(series_metadata),
            'validation_status': 'PENDING',
        }

        # Filter out scout/localizer images
        main_series_metadata = [ds for ds in series_metadata if 'ORIGINAL' in getattr(ds, 'ImageType', [])]
        
        scout_slice_count = len(series_metadata) - len(main_series_metadata)
        report['scout_slice_count'] = scout_slice_count
        report['total_dcms'] = len(main_series_metadata)

        if not main_series_metadata:
            report['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
            series_reports.append(report)
            continue

        # Use a sample for common metadata
        sample_ds = main_series_metadata[0]
        iop = getattr(sample_ds, 'ImageOrientationPatient', None)
        rows = getattr(sample_ds, 'Rows', 'N/A')
        cols = getattr(sample_ds, 'Columns', 'N/A')
        report['orientation'] = get_orientation(iop)
        report['modality'] = getattr(sample_ds, 'Modality', 'N/A')
        report['slice_thickness'] = getattr(sample_ds, 'SliceThickness', 'N/A')
        report['patient_age'] = getattr(sample_ds, 'PatientAge', 'N/A')
        report['patient_sex'] = getattr(sample_ds, 'PatientSex', 'N/A')
        report['exam_date'] = getattr(sample_ds, 'StudyDate', 'N/A')
        report['image_dimensions'] = f"{rows}x{cols}"

        # Projection-Based Sorting
        try:
            row_vec = np.array(iop[:3])
            col_vec = np.array(iop[3:])
            normal_vec = np.cross(row_vec, col_vec)

            for ds in main_series_metadata:
                ipp = np.array(getattr(ds, 'ImagePositionPatient', [0,0,0]))
                ds.dist = np.dot(ipp, normal_vec)
            
            main_series_metadata.sort(key=lambda x: x.dist)

            # Spacing Analysis
            distances = np.array([ds.dist for ds in main_series_metadata])
            deltas = np.abs(np.diff(distances))
            unique_spacings = np.unique(np.around(deltas, decimals=2))

            num_duplicates = np.sum(deltas < 0.001)
            report['duplicate_slice_count'] = int(num_duplicates)

            if num_duplicates > 0:
                report['validation_status'] = 'DUPLICATE_SLICES'
            elif len(unique_spacings) == 1:
                report['validation_status'] = 'OK' # Simplified from original for clarity
            elif len(unique_spacings) > 1:
                report['validation_status'] = 'VARIABLE_SPACING' # Could be gapped or truly variable

        except Exception as e:
            report['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

        series_reports.append(report)

    return series_reports

def main():
    """
    Analyzes all folders flagged with 'MIXED_SERIES_ERROR'
    in a given data map CSV.
    """
    try:
        with open(INPUT_CSV_PATH, 'r', newline='') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at '{INPUT_CSV_PATH}'")
        return

    mixed_series_cases = [
        row for row in all_rows
        if row.get('validation_status', '').startswith('MIXED_SERIES_ERROR')
    ]

    if not mixed_series_cases:
        print("No cases with 'MIXED_SERIES_ERROR' found in the input CSV.")
        return

    print(f"Found {len(mixed_series_cases)} cases with mixed series. Analyzing...")

    all_series_reports = []
    for case in mixed_series_cases:
        folder_path = os.path.join(BASE_DATA_DIR, case['data_path'])
        reports = analyze_mixed_series(folder_path)
        # Add the parent case name to each sub-series report for context
        for report in reports:
            report['parent_case_name'] = case['fixed_name']
        all_series_reports.extend(reports)

    if all_series_reports:
        # Collect all unique keys from all report dictionaries to build a complete header.
        # This prevents errors if the first report is missing optional keys.
        all_keys = set()
        for report in all_series_reports:
            all_keys.update(report.keys())
        
        # Ensure 'parent_case_name' is the first column for readability.
        fieldnames = sorted(list(all_keys))
        if 'parent_case_name' in fieldnames:
            fieldnames.remove('parent_case_name')
            fieldnames.insert(0, 'parent_case_name')

        with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_series_reports)
        print(f"\nAnalysis complete. Consolidated report saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()