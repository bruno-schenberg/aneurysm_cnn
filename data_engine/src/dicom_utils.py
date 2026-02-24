import logging
logger = logging.getLogger("dicom_ingestion")
import os
import csv
from collections import defaultdict
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np

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

def load_dicom_metadata(path):
    """
    Loads all DICOM file headers from a given directory path.

    Args:
        path (str): The full path to the directory containing .dcm files.

    Returns:
        list: A list of pydicom dataset objects, one for each DICOM file.
    """
    metadata_list = []
    for f in os.listdir(path):
        if f.lower().endswith('.dcm'):
            try:
                dcm_path = os.path.join(path, f)
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                metadata_list.append(ds)
            except InvalidDicomError:
                logger.warning(f"  - Warning: Skipping invalid DICOM file: {dcm_path}")
    return metadata_list

def validate_dcms(data_mapping, base_path):
    """
    Validates DICOM series using projection-based sorting and integrity checks.

    Updates each item in data_mapping with validation status and DICOM metadata.
    """
    logger.info("\nValidating DICOM series with Projection-Based Sorting...")

    for item in data_mapping:
        data_path = item.get('data_path')
        if data_path in ('EMPTY', 'MISSING', 'DUPLICATE_DATA'):
            item['validation_status'] = 'NOT_APPLICABLE'
            continue

        logger.info(f"  - Validating {item['fixed_name']}...")
        full_path = os.path.join(base_path, data_path)

        try:
            # 1. Load DICOM metadata from the specified path.
            metadata_list = load_dicom_metadata(full_path)

            if not metadata_list:
                item['validation_status'] = 'NO_DCM_FILES'
                continue

            # 2. Check for Mixed Series (SeriesInstanceUID)
            uids = {ds.SeriesInstanceUID for ds in metadata_list}
            if len(uids) > 1:
                item['validation_status'] = f'MIXED_SERIES_ERROR ({len(uids)} UIDs)'
                continue
            
            # --- Filter out scout/localizer images based on ImageType ---
            # The main series will have 'ORIGINAL' in its ImageType.
            # Scout/localizer images often have 'DERIVED'.
            main_series_metadata = []
            for ds in metadata_list:
                image_type = getattr(ds, 'ImageType', [])
                if 'ORIGINAL' in image_type:
                    main_series_metadata.append(ds)
            
            if not main_series_metadata:
                item['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
                continue

            # Update total_dcms to reflect the main series size and count scouts
            scout_slice_count = len(metadata_list) - len(main_series_metadata)
            item['total_dcms'] = len(main_series_metadata)
            item['scout_slice_count'] = scout_slice_count

            # 3. Projection-Based Sorting
            sample_ds = main_series_metadata[0]
            iop = getattr(sample_ds, 'ImageOrientationPatient', None)
            ipp_sample = getattr(sample_ds, 'ImagePositionPatient', None)
            pixel_spacing = getattr(sample_ds, 'PixelSpacing', None)
            rows = getattr(sample_ds, 'Rows', None)
            cols = getattr(sample_ds, 'Columns', None)

            if iop is None or ipp_sample is None or pixel_spacing is None or rows is None or cols is None:
                item['validation_status'] = 'MISSING_MANDATORY_SPATIAL_TAGS'
                continue

            row_vec = np.array(iop[:3])
            col_vec = np.array(iop[3:])
            normal_vec = np.cross(row_vec, col_vec)

            for ds in main_series_metadata:
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is None:
                    ds.dist = 0 # Should not happen if ipp_sample passed, but being safe
                else:
                    ds.dist = np.dot(np.array(ipp), normal_vec)

            main_series_metadata.sort(key=lambda x: x.dist)

            # 4. Spacing Analysis
            distances = np.array([ds.dist for ds in main_series_metadata])
            deltas = np.abs(np.diff(distances))
            unique_spacings = np.unique(np.around(deltas, decimals=2))

            # 5. Update item metadata
            rows = getattr(sample_ds, 'Rows', 'N/A')
            cols = getattr(sample_ds, 'Columns', 'N/A')
            
            # Calculate effective slice thickness/spacing from projection distances
            effective_slice_thickness = 'N/A'
            if len(deltas) > 0:
                # Use the most common spacing as the representative value
                vals, counts = np.unique(np.around(deltas, decimals=2), return_counts=True)
                effective_slice_thickness = vals[np.argmax(counts)]
            elif len(main_series_metadata) == 1:
                # If only one slice, fall back to the DICOM tag
                effective_slice_thickness = getattr(sample_ds, 'SliceThickness', 'N/A')

            item['orientation'] = get_orientation(iop)
            item['modality'] = getattr(sample_ds, 'Modality', 'N/A')
            item['slice_thickness'] = effective_slice_thickness
            item['patient_sex'] = getattr(sample_ds, 'PatientSex', 'N/A')
            item['image_dimensions'] = f"{rows}x{cols}"

            # Calculate exam_size (total coverage)
            exam_size = 'N/A'
            # Ensure slice thickness is a number before multiplying
            if isinstance(effective_slice_thickness, (int, float, np.number)):
                exam_size = item['total_dcms'] * effective_slice_thickness
            item['exam_size'] = exam_size

            num_duplicates = np.sum(deltas < 0.001)
            item['duplicate_slice_count'] = int(num_duplicates)

            if num_duplicates > 0:
                item['validation_status'] = f'DUPLICATE_SLICES'
            else:
                # Refined check for gapped vs. variable spacing
                if len(unique_spacings) == 1:
                    # Perfectly uniform spacing, check instance numbers for completeness
                    instance_numbers = sorted([int(getattr(ds, 'InstanceNumber', 0)) for ds in main_series_metadata])
                    is_gapped_instance = any(np.diff(instance_numbers) != 1)
                    item['validation_status'] = 'GAPPED_SEQUENCE' if is_gapped_instance else 'OK'
                elif len(unique_spacings) == 2:
                    # Check if one spacing is roughly double the other, indicating missing slices
                    sorted_spacings = sorted(unique_spacings)
                    if np.isclose(sorted_spacings[1], 2 * sorted_spacings[0], atol=0.1):
                         # Count how many "double gaps" exist
                        gaps = np.sum(np.isclose(deltas, sorted_spacings[1], atol=0.1))
                        item['validation_status'] = f'GAPPED_SEQUENCE ({gaps} missing)'
                    else:
                        item['validation_status'] = 'VARIABLE_SPACING'
                elif len(unique_spacings) > 2:
                    # More than two spacing values is truly variable
                    item['validation_status'] = 'VARIABLE_SPACING'

        except Exception as e:
            item['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

    # After initial validation, perform outlier detection on 'OK' series
    data_mapping = validate_dcm_count(data_mapping)

    logger.info("Validation complete.")
    return data_mapping

def validate_dcm_count(data_mapping):
    """
    Identifies outliers in exam size for validated series using the IQR method.

    Calculates quartiles for exam sizes of 'OK' series and flags those
    outside the fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

    Args:
        data_mapping (list): The list of dictionaries, where each dictionary
                             represents a DICOM series and its metadata.

    Returns:
        list: The updated data_mapping with outlier statuses.
    """
    ok_exams = [item for item in data_mapping if item.get('validation_status') == 'OK']
    
    # Safely convert exam_size to float for calculation, skipping non-numeric values.
    # This mirrors the logic from the successful debug script.
    exam_sizes = []
    for item in ok_exams:
        try:
            exam_sizes.append(float(item['exam_size']))
        except (ValueError, TypeError):
            continue # Skip if exam_size is 'N/A' or otherwise not convertible

    if not exam_sizes:
        return data_mapping

    q1 = np.percentile(exam_sizes, 25)
    q3 = np.percentile(exam_sizes, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    logger.info("\nAnalyzing exam size distribution for 'OK' series...")
    logger.info(f"  - Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    logger.info(f"  - Lower Fence: {lower_fence:.2f}, Upper Fence: {upper_fence:.2f}")

    for item in ok_exams:
        if isinstance(item.get('exam_size'), (int, float, np.number)):
            if item['exam_size'] < lower_fence:
                item['validation_status'] = 'BELOW_LIMIT'
            elif item['exam_size'] > upper_fence:
                item['validation_status'] = 'ABOVE_LIMIT'

    outliers_below = [item for item in ok_exams if item.get('validation_status') == 'BELOW_LIMIT']
    outliers_above = [item for item in ok_exams if item.get('validation_status') == 'ABOVE_LIMIT']

    if outliers_below:
        logger.info(f"\nFound {len(outliers_below)} exams BELOW the lower fence:")
        for item in outliers_below:
            logger.info(f"  - {item['fixed_name']} (Exam Size: {item['exam_size']:.2f})")

    if outliers_above:
        logger.info(f"\nFound {len(outliers_above)} exams ABOVE the upper fence:")
        for item in outliers_above:
            logger.info(f"  - {item['fixed_name']} (Exam Size: {item['exam_size']:.2f})")

    return data_mapping

def analyze_mixed_series(folder_path):
    """
    Analyzes a directory with mixed DICOM series by grouping files by SeriesInstanceUID
    and running validation on each individual series.
    """
    if not os.path.isdir(folder_path):
        return []

    all_metadata = load_dicom_metadata(folder_path)
    if not all_metadata:
        return []

    series_groups = defaultdict(list)
    for ds in all_metadata:
        series_groups[ds.SeriesInstanceUID].append(ds)

    series_reports = []
    for i, (uid, series_metadata) in enumerate(series_groups.items()):
        report = {
            'series_index': i + 1,
            'series_instance_uid': uid,
            'total_dcms': len(series_metadata),
            'validation_status': 'PENDING',
        }

        main_series_metadata = [ds for ds in series_metadata if 'ORIGINAL' in getattr(ds, 'ImageType', [])]
        
        scout_slice_count = len(series_metadata) - len(main_series_metadata)
        report['scout_slice_count'] = scout_slice_count
        report['total_dcms'] = len(main_series_metadata)

        if not main_series_metadata:
            report['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
            series_reports.append(report)
            continue

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

        try:
            row_vec = np.array(iop[:3])
            col_vec = np.array(iop[3:])
            normal_vec = np.cross(row_vec, col_vec)

            for ds in main_series_metadata:
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is None:
                    ds.dist = 0
                else:
                    ds.dist = np.dot(np.array(ipp), normal_vec)
            
            main_series_metadata.sort(key=lambda x: x.dist)

            distances = np.array([ds.dist for ds in main_series_metadata])
            deltas = np.abs(np.diff(distances))
            unique_spacings = np.unique(np.around(deltas, decimals=2))

            num_duplicates = np.sum(deltas < 0.001)
            report['duplicate_slice_count'] = int(num_duplicates)

            if num_duplicates > 0:
                report['validation_status'] = 'DUPLICATE_SLICES'
            elif len(unique_spacings) == 1:
                report['validation_status'] = 'OK'
            elif len(unique_spacings) > 1:
                report['validation_status'] = 'VARIABLE_SPACING'

        except Exception as e:
            report['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

        series_reports.append(report)

    return series_reports

def analyze_mixed_folders(data_mapping, base_path, output_csv_path):
    """
    Analyzes folders flagged with MIXED_SERIES_ERROR and outputs a report.
    """
    mixed_cases = [item for item in data_mapping if str(item.get('validation_status')).startswith('MIXED_SERIES_ERROR')]
    if not mixed_cases:
        return

    logger.info(f"Analyzing {len(mixed_cases)} cases with mixed series...")
    all_series_reports = []
    
    for case in mixed_cases:
        folder_path = os.path.join(base_path, case['data_path'])
        reports = analyze_mixed_series(folder_path)
        for report in reports:
            report['parent_case_name'] = case['fixed_name']
        all_series_reports.extend(reports)

    if all_series_reports:
        all_keys = set()
        for report in all_series_reports:
            all_keys.update(report.keys())
        
        fieldnames = sorted(list(all_keys))
        if 'parent_case_name' in fieldnames:
            fieldnames.remove('parent_case_name')
            fieldnames.insert(0, 'parent_case_name')

        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_series_reports)
        logger.info(f"Mixed series analysis complete. Saved to: {output_csv_path}")