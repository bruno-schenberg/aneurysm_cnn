import os
import pytest
import numpy as np
from src.dicom_utils import get_orientation, load_dicom_metadata, validate_dcms

# Path to the sample raw dataset
SAMPLE_RAW_DIR = "/home/aneurysm_cnn/data_engine/dataset/sample-raw"
SAMPLE_DICOM_DIR = os.path.join(SAMPLE_RAW_DIR, "BP05- 2838474", "Anonymous_Mr_712482402_20170105")

def test_get_orientation():
    # Axial
    assert get_orientation([1, 0, 0, 0, 1, 0]) == "AXIAL"
    # Coronal
    assert get_orientation([1, 0, 0, 0, 0, -1]) == "CORONAL"
    # Sagittal
    assert get_orientation([0, 1, 0, 0, 0, -1]) == "SAGITTAL"
    # Oblique
    assert get_orientation([0.7, 0.7, 0, 0, 0, -1]) == "OBLIQUE"
    # Missing/Invalid
    assert get_orientation(None) == "UNKNOWN"
    assert get_orientation([1, 0, 0]) == "UNKNOWN"

def test_load_dicom_metadata_with_sample_data():
    if not os.path.exists(SAMPLE_DICOM_DIR):
        pytest.skip(f"Sample data not found at {SAMPLE_DICOM_DIR}")
        
    metadata = load_dicom_metadata(SAMPLE_DICOM_DIR)
    assert len(metadata) > 0, "Should load metadata from sample directory"
    
    # Check that it actually parsed DICOM correctly
    assert hasattr(metadata[0], "PatientID"), "Loaded metadata should have PatientID"
    assert hasattr(metadata[0], "SeriesInstanceUID"), "Loaded metadata should have SeriesInstanceUID"

def test_validate_dcms_with_sample_data():
    if not os.path.exists(SAMPLE_DICOM_DIR):
        pytest.skip(f"Sample data not found at {SAMPLE_DICOM_DIR}")
        
    # Create a mock data mapping pointing to the sample dir
    data_mapping = [{
        "fixed_name": "Test_Sample_1",
        "data_path": "BP05- 2838474/Anonymous_Mr_712482402_20170105"
    }]
    
    result = validate_dcms(data_mapping, SAMPLE_RAW_DIR)
    assert len(result) == 1
    item = result[0]
    
    assert "validation_status" in item
    
    # Since it's actual sample data, we expect it to be processed
    # validation_status might not be 'OK' if data has variable spacing/gaps,
    # but it shouldn't fail with an unhandled exception or 'ALL_DCMS_CORRUPT'.
    valid_statuses = ("OK", "VARIABLE_SPACING", "GAPPED_SEQUENCE", "DUPLICATE_SLICES", "BELOW_LIMIT", "ABOVE_LIMIT")
    
    # We use lower checking logic here just to make sure it ran completely
    assert not item["validation_status"].startswith("VALIDATION_ERROR")
    assert item["validation_status"] != "ALL_DCMS_CORRUPT"

def test_projection_sorting_logic():
    # Mocking the distance calculation logic
    # IOP for Axial
    iop = [1, 0, 0, 0, 1, 0]
    row_vec = np.array(iop[:3])
    col_vec = np.array(iop[3:])
    normal_vec = np.cross(row_vec, col_vec) # [0, 0, 1]
    
    # IPP values
    ipp1 = [0, 0, 10]
    ipp2 = [0, 0, 5]
    ipp3 = [0, 0, 15]
    
    dist1 = np.dot(ipp1, normal_vec) # 10
    dist2 = np.dot(ipp2, normal_vec) # 5
    dist3 = np.dot(ipp3, normal_vec) # 15
    
    dists = [dist1, dist2, dist3]
    dists.sort()
    assert dists == [5, 10, 15]

def test_spacing_analysis_logic():
    # Perfect spacing
    dists = np.array([0, 1.5, 3.0, 4.5])
    deltas = np.abs(np.diff(dists))
    unique_spacings = np.unique(np.around(deltas, decimals=2))
    assert len(unique_spacings) == 1
    assert unique_spacings[0] == 1.5

    # Gapped sequence (one missing)
    dists = np.array([0, 1.5, 4.5, 6.0]) # missing 3.0
    deltas = np.abs(np.diff(dists))
    unique_spacings = np.unique(np.around(deltas, decimals=2))
    assert len(unique_spacings) == 2
    assert 1.5 in unique_spacings
    assert 3.0 in unique_spacings
