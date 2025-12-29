import os
import csv
import argparse
from utils.dicom_utils import analyze_mixed_series

def main():
    """
    A command-line tool to analyze all folders flagged with 'MIXED_SERIES_ERROR'
    in a given data map CSV.
    """
    parser = argparse.ArgumentParser(
        description="Find and analyze all mixed DICOM series from a data map CSV, generating a consolidated report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input data map CSV (e.g., folder_rename_map.csv)."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="The base directory where the data folders are located (e.g., /mnt/data/cases-3/raw)."
    )
    parser.add_argument(
        "-o", "--output",
        default="mixed_series_analysis.csv",
        help="Path for the consolidated output CSV report."
    )
    args = parser.parse_args()

    try:
        with open(args.input_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at '{args.input_csv}'")
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
        folder_path = os.path.join(args.base_dir, case['data_path'])
        reports = analyze_mixed_series(folder_path)
        # Add the parent case name to each sub-series report for context
        for report in reports:
            report['parent_case_name'] = case['fixed_name']
        all_series_reports.extend(reports)

    if all_series_reports:
        # Ensure 'parent_case_name' is the first column for clarity
        fieldnames = ['parent_case_name'] + [k for k in all_series_reports[0].keys() if k != 'parent_case_name']
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_series_reports)
        print(f"\nAnalysis complete. Consolidated report saved to: {args.output}")

if __name__ == "__main__":
    main()