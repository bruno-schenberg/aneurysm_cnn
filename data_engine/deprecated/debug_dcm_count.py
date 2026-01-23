import csv
import numpy as np

INPUT_CSV_PATH = "folder_rename_map.csv"

def analyze_exam_sizes_from_csv(csv_path):
    """
    Loads data from the output CSV, re-calculates outlier fences for 'OK'
    exams, and identifies which exams fall outside these limits.

    This serves as a debugging tool to verify the quartile calculations on
    the final, static dataset.

    Args:
        csv_path (str): The path to the folder_rename_map.csv file.
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_exams = list(reader)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # 1. Filter for 'OK' exams and safely convert 'exam_size' to float
    ok_exams = []
    for exam in all_exams:
        if exam.get('validation_status') == 'OK':
            try:
                # Convert exam_size to float for calculation
                exam['exam_size_numeric'] = float(exam['exam_size'])
                ok_exams.append(exam)
            except (ValueError, TypeError):
                # Skip exams where exam_size is 'N/A' or cannot be converted
                continue

    exam_sizes = [exam['exam_size_numeric'] for exam in ok_exams]

    if not exam_sizes:
        print("No 'OK' exams with valid numeric 'exam_size' found in the CSV.")
        return

    # 2. Calculate quartiles and fences using NumPy
    q1 = np.percentile(exam_sizes, 25)
    q3 = np.percentile(exam_sizes, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    print(f"Analysis of 'OK' exams from '{csv_path}':")
    print("-" * 40)
    print(f"Total 'OK' exams with numeric size: {len(exam_sizes)}")
    print(f"  - Q1 (25th percentile): {q1:.2f}")
    print(f"  - Q3 (75th percentile): {q3:.2f}")
    print(f"  - IQR (Q3 - Q1):        {iqr:.2f}")
    print(f"  - Lower Fence:          {lower_fence:.2f}")
    print(f"  - Upper Fence:          {upper_fence:.2f}")
    print("-" * 40)

    # 3. Identify and list the outliers
    outliers_below = [e for e in ok_exams if e['exam_size_numeric'] < lower_fence]
    outliers_above = [e for e in ok_exams if e['exam_size_numeric'] > upper_fence]

    print(f"\nFound {len(outliers_below)} exams BELOW the lower fence:")
    for exam in outliers_below:
        print(f"  - {exam['fixed_name']} (Exam Size: {exam['exam_size_numeric']:.2f})")

    print(f"\nFound {len(outliers_above)} exams ABOVE the upper fence:")
    for exam in outliers_above:
        print(f"  - {exam['fixed_name']} (Exam Size: {exam['exam_size_numeric']:.2f})")

if __name__ == "__main__":
    analyze_exam_sizes_from_csv(INPUT_CSV_PATH)