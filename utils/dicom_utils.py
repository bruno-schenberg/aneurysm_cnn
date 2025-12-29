import os
import pydicom
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
            # 1. Load headers and basic series check
            metadata_list = []
            for f in os.listdir(full_path):
                if f.lower().endswith('.dcm'):
                    ds = pydicom.dcmread(os.path.join(full_path, f), stop_before_pixels=True)
                    metadata_list.append(ds)

            if not metadata_list:
                item['validation_status'] = 'NO_DCM_FILES'
                continue

            # 2. Check for Mixed Series (SeriesInstanceUID)
            uids = {ds.SeriesInstanceUID for ds in metadata_list}
            if len(uids) > 1:
                item['validation_status'] = f'MIXED_SERIES_ERROR ({len(uids)} UIDs)'
                continue

            # 3. Projection-Based Sorting
            sample_ds = metadata_list[0]
            iop = sample_ds.ImageOrientationPatient
            row_vec = np.array(iop[:3])
            col_vec = np.array(iop[3:])
            normal_vec = np.cross(row_vec, col_vec)

            for ds in metadata_list:
                ipp = np.array(ds.ImagePositionPatient)
                ds.dist = np.dot(ipp, normal_vec)
            
            metadata_list.sort(key=lambda x: x.dist)

            # 4. Spacing Analysis
            distances = np.array([ds.dist for ds in metadata_list])
            deltas = np.abs(np.diff(distances))
            unique_spacings = np.unique(np.around(deltas, decimals=2))
            
            # 5. Update item metadata
            item['orientation'] = get_orientation(iop)
            item['series_description'] = getattr(sample_ds, 'SeriesDescription', 'N/A')
            item['modality'] = getattr(sample_ds, 'Modality', 'N/A')
            item['slice_thickness'] = getattr(sample_ds, 'SliceThickness', 'N/A')
            item['spacing_type'] = "CONSISTENT" if len(unique_spacings) <= 1 else "VARIABLE"

            if np.any(deltas < 0.001):
                item['validation_status'] = 'DUPLICATE_SLICES'
            elif len(unique_spacings) > 1:
                item['validation_status'] = 'VARIABLE_SPACING'
            else:
                instance_numbers = sorted([int(ds.InstanceNumber) for ds in metadata_list])
                item['validation_status'] = 'OK' if all(np.diff(instance_numbers) == 1) else 'GAPPED_SEQUENCE'

        except Exception as e:
            item['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

    print("Validation complete.")
    return data_mapping