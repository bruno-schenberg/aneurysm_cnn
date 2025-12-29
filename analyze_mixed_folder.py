import os
import csv
from utils.dicom_utils import analyze_mixed_series

# --- Configuration ---
INPUT_CSV_PATH = "folder_rename_map.csv"
BASE_DATA_DIR = "/mnt/data/cases-3/raw"
OUTPUT_CSV_PATH = "mixed_series_analysis.csv"
# -------------------

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