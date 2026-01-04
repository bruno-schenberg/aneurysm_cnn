import os
import pydicom
from pydicom.errors import InvalidDicomError
from collections import defaultdict
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
                print(f"  - Warning: Skipping invalid DICOM file: {dcm_path}")
    return metadata_list

def validate_dcms(data_mapping, base_path):
    """
    Validates DICOM series using projection-based sorting and integrity checks.

    Updates each item in data_mapping with validation status and DICOM metadata.
    """
    print("\nValidating DICOM series with Projection-Based Sorting...")

    for item in data_mapping:
        data_path = item.get('data_path')
        if data_path in ('EMPTY', 'MISSING', 'DUPLICATE_DATA'):
            item['validation_status'] = 'NOT_APPLICABLE'
            continue

        print(f"  - Validating {item['fixed_name']}...")
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
            row_vec = np.array(iop[:3])
            col_vec = np.array(iop[3:])
            normal_vec = np.cross(row_vec, col_vec)

            for ds in main_series_metadata:
                ipp = np.array(getattr(ds, 'ImagePositionPatient', [0,0,0]))
                ds.dist = np.dot(ipp, normal_vec)

            main_series_metadata.sort(key=lambda x: x.dist)

            # 4. Spacing Analysis
            distances = np.array([ds.dist for ds in main_series_metadata])
            deltas = np.abs(np.diff(distances))
            unique_spacings = np.unique(np.around(deltas, decimals=2))

            # 5. Update item metadata
            item['orientation'] = get_orientation(iop)
            item['series_description'] = getattr(sample_ds, 'SeriesDescription', 'N/A')
            item['modality'] = getattr(sample_ds, 'Modality', 'N/A')
            item['slice_thickness'] = getattr(sample_ds, 'SliceThickness', 'N/A')
            item['patient_age'] = getattr(sample_ds, 'PatientAge', 'N/A')
            item['patient_sex'] = getattr(sample_ds, 'PatientSex', 'N/A')

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

    print("Validation complete.")
    return data_mapping

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