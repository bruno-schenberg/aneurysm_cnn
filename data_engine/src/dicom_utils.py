import logging
import os
import csv
from collections import defaultdict
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np

logger = logging.getLogger("dicom_ingestion")

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
    Loads all valid DICOM file headers from a given directory path.
    Skips files that are not valid DICOMs.
    """
    metadata_list = []
    for f in os.listdir(path):
        if f.lower().endswith('.dcm'):
            try:
                dcm_path = os.path.join(path, f)
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                metadata_list.append(ds)
            except InvalidDicomError:
                logger.warning(f"  - Warning: Skipping corrupt/invalid DICOM file: {dcm_path}")
    return metadata_list

def _calculate_projection_distances(metadata_list, iop):
    """
    Sorts a list of DICOM metadata objects in 3D space by projecting their
    ImagePositionPatient onto a normal vector derived from ImageOrientationPatient.
    """
    row_vec = np.array(iop[:3])
    col_vec = np.array(iop[3:])
    normal_vec = np.cross(row_vec, col_vec)

    for ds in metadata_list:
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp is None:
            ds.dist = 0 # Failsafe, shouldn't happen if validation passed earlier
        else:
            ds.dist = np.dot(np.array(ipp), normal_vec)

    # Sort in place based on the calculated physical distance
    metadata_list.sort(key=lambda x: x.dist)

def _evaluate_spacing(metadata_list):
    """
    Analyzes the physical distance between consecutive slices to determine
    if the sequence is uniform, gapped, or has variable spacing.
    Returns: (validation_status, unique_spacings, duplicate_count, deltas)
    """
    distances = np.array([ds.dist for ds in metadata_list])
    deltas = np.abs(np.diff(distances))
    unique_spacings = np.unique(np.around(deltas, decimals=2))

    num_duplicates = int(np.sum(deltas < 0.001))

    if num_duplicates > 0:
        return 'DUPLICATE_SLICES', unique_spacings, num_duplicates, deltas

    if len(unique_spacings) == 1:
        # Perfectly uniform spacing, check instance numbers for completeness
        instance_numbers = sorted([int(getattr(ds, 'InstanceNumber', 0)) for ds in metadata_list])
        is_gapped_instance = any(np.diff(instance_numbers) != 1)
        return ('GAPPED_SEQUENCE' if is_gapped_instance else 'OK'), unique_spacings, 0, deltas

    elif len(unique_spacings) == 2:
        # Check if one spacing is roughly double the other, indicating missing slices
        sorted_spacings = sorted(unique_spacings)
        if np.isclose(sorted_spacings[1], 2 * sorted_spacings[0], atol=0.1):
            gaps = np.sum(np.isclose(deltas, sorted_spacings[1], atol=0.1))
            return f'GAPPED_SEQUENCE ({gaps} missing)', unique_spacings, 0, deltas
        else:
            return 'VARIABLE_SPACING', unique_spacings, 0, deltas

    else:
        # More than two spacing values
        return 'VARIABLE_SPACING', unique_spacings, 0, deltas

def validate_dcms(data_mapping, base_path):
    """
    Validates DICOM series using projection-based sorting and integrity checks.
    Updates each item in data_mapping with validation status and metadata.
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
            # 1. Load DICOM metadata
            metadata_list = load_dicom_metadata(full_path)

            if not metadata_list:
                item['validation_status'] = 'ALL_DCMS_CORRUPT'
                continue

            # 2. Check for Mixed Series (Multiple scans dumped in one folder)
            uids = {ds.SeriesInstanceUID for ds in metadata_list}
            if len(uids) > 1:
                item['validation_status'] = f'MIXED_SERIES_ERROR ({len(uids)} UIDs)'
                continue
            
            # 3. Filter out scout/localizer images (Keep only 'ORIGINAL')
            main_series_metadata = [ds for ds in metadata_list if 'ORIGINAL' in getattr(ds, 'ImageType', [])]
            
            if not main_series_metadata:
                item['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
                continue

            item['total_dcms'] = len(main_series_metadata)
            item['scout_slice_count'] = len(metadata_list) - len(main_series_metadata)

            # 4. Extract Spatial Tags
            sample_ds = main_series_metadata[0]
            iop = getattr(sample_ds, 'ImageOrientationPatient', None)
            ipp = getattr(sample_ds, 'ImagePositionPatient', None)
            
            if iop is None or ipp is None or getattr(sample_ds, 'PixelSpacing', None) is None:
                item['validation_status'] = 'MISSING_MANDATORY_SPATIAL_TAGS'
                continue

            # 5. Math: Projection-Based Sorting
            _calculate_projection_distances(main_series_metadata, iop)

            # 6. Math: Spacing Analysis & Quality Control
            status, unique_spacings, duplicates, deltas = _evaluate_spacing(main_series_metadata)
            item['validation_status'] = status
            item['duplicate_slice_count'] = duplicates

            # 7. Extract final item metadata
            rows = getattr(sample_ds, 'Rows', 'N/A')
            cols = getattr(sample_ds, 'Columns', 'N/A')
            
            # Calculate effective slice thickness based on real distances
            if len(deltas) > 0:
                vals, counts = np.unique(np.around(deltas, decimals=2), return_counts=True)
                effective_slice_thickness = vals[np.argmax(counts)]
            elif len(main_series_metadata) == 1:
                effective_slice_thickness = getattr(sample_ds, 'SliceThickness', 'N/A')
            else:
                effective_slice_thickness = 'N/A'

            item['orientation'] = get_orientation(iop)
            item['modality'] = getattr(sample_ds, 'Modality', 'N/A')
            item['slice_thickness'] = effective_slice_thickness
            item['patient_sex'] = getattr(sample_ds, 'PatientSex', 'N/A')
            item['image_dimensions'] = f"{rows}x{cols}"

            if isinstance(effective_slice_thickness, (int, float, np.number)):
                item['exam_size'] = item['total_dcms'] * effective_slice_thickness
            else:
                item['exam_size'] = 'N/A'

        except Exception as e:
            item['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

    # 8. Perform statistical outlier detection on 'OK' series sizes
    data_mapping = validate_dcm_count(data_mapping)

    logger.info("Validation complete.")
    return data_mapping

def validate_dcm_count(data_mapping):
    """
    Identifies outliers in exam size for validated series using the IQR method.
    Flags those outside the fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    """
    ok_exams = [item for item in data_mapping if item.get('validation_status') == 'OK']
    
    exam_sizes = []
    for item in ok_exams:
        try:
            exam_sizes.append(float(item['exam_size']))
        except (ValueError, TypeError):
            continue 

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
    and running a mini-validation on each individual series.
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
        report['scout_slice_count'] = len(series_metadata) - len(main_series_metadata)
        report['total_dcms'] = len(main_series_metadata)

        if not main_series_metadata:
            report['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
            series_reports.append(report)
            continue

        sample_ds = main_series_metadata[0]
        iop = getattr(sample_ds, 'ImageOrientationPatient', None)
        
        report['orientation'] = get_orientation(iop)
        report['modality'] = getattr(sample_ds, 'Modality', 'N/A')
        report['slice_thickness'] = getattr(sample_ds, 'SliceThickness', 'N/A')
        report['patient_age'] = getattr(sample_ds, 'PatientAge', 'N/A')
        report['patient_sex'] = getattr(sample_ds, 'PatientSex', 'N/A')
        report['exam_date'] = getattr(sample_ds, 'StudyDate', 'N/A')
        report['image_dimensions'] = f"{getattr(sample_ds, 'Rows', 'N/A')}x{getattr(sample_ds, 'Columns', 'N/A')}"

        try:
            if iop is not None and getattr(sample_ds, 'ImagePositionPatient', None) is not None:
                _calculate_projection_distances(main_series_metadata, iop)
                status, _, duplicates, _ = _evaluate_spacing(main_series_metadata)
                report['validation_status'] = status
                report['duplicate_slice_count'] = duplicates
            else:
                 report['validation_status'] = 'MISSING_MANDATORY_SPATIAL_TAGS'
                 
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
