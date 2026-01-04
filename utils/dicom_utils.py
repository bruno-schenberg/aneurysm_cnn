import os
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