import os
import pytest
import numpy as np
from src.dicom_utils import (
    _evaluate_spacing,
    _flag_exam_size_outliers,
    get_orientation,
    load_dicom_metadata,
    validate_dcms,
)

# Path to the sample raw dataset
SAMPLE_RAW_DIR = "/mnt/data/cases-3/sample-raw"
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


# ---------------------------------------------------------------------------
# _evaluate_spacing — branch coverage via mock datasets
# ---------------------------------------------------------------------------


class _MockDS:
    """Minimal stand-in for a pydicom Dataset that _evaluate_spacing reads."""
    def __init__(self, dist: float, instance_number: int = 1):
        self.dist = dist
        self.InstanceNumber = instance_number


def test_evaluate_spacing_duplicate_slices():
    """Two slices at the same position must return DUPLICATE_SLICES."""
    ds_list = [_MockDS(0.0, 1), _MockDS(0.0, 2), _MockDS(1.5, 3)]
    status, _, dupes, _ = _evaluate_spacing(ds_list)
    assert status == "DUPLICATE_SLICES"
    assert dupes > 0


def test_evaluate_spacing_ok_uniform_spacing():
    """Perfectly uniform spacing with consecutive instance numbers must return OK."""
    ds_list = [_MockDS(d, i + 1) for i, d in enumerate([0.0, 1.5, 3.0, 4.5])]
    status, _, _, _ = _evaluate_spacing(ds_list)
    assert status == "OK"


def test_evaluate_spacing_gapped_instance_numbers():
    """Uniform spatial spacing but non-consecutive instance numbers must return GAPPED_SEQUENCE."""
    # Physical distances are evenly spaced; instance numbers skip from 2 to 4.
    ds_list = [_MockDS(d, n) for d, n in zip([0.0, 1.5, 3.0, 4.5], [1, 2, 4, 5])]
    status, _, _, _ = _evaluate_spacing(ds_list)
    assert status == "GAPPED_SEQUENCE"


def test_evaluate_spacing_variable_two_non_doubled_spacings():
    """Two unrelated spacings (not a 2× ratio) must return VARIABLE_SPACING."""
    # Deltas: 1.5, 3.3 — ratio is 2.2, not 2.0 within tolerance.
    ds_list = [_MockDS(d, i + 1) for i, d in enumerate([0.0, 1.5, 4.8])]
    status, _, _, _ = _evaluate_spacing(ds_list)
    assert status == "VARIABLE_SPACING"


def test_evaluate_spacing_more_than_two_spacings():
    """More than two distinct spacings must return VARIABLE_SPACING."""
    # Deltas: 1.0, 1.5, 2.0 — three unique values.
    ds_list = [_MockDS(d, i + 1) for i, d in enumerate([0.0, 1.0, 2.5, 4.5])]
    status, _, _, _ = _evaluate_spacing(ds_list)
    assert status == "VARIABLE_SPACING"


# ---------------------------------------------------------------------------
# validate_dcms — branches that do not require real DICOM files
# ---------------------------------------------------------------------------


def test_validate_dcms_not_applicable_for_special_paths():
    """Items with EMPTY, MISSING, or DUPLICATE_DATA data_path must be skipped."""
    for code in ("EMPTY", "MISSING", "DUPLICATE_DATA"):
        data_mapping = [{"fixed_name": "BP001", "data_path": code}]
        result = validate_dcms(data_mapping, "/nonexistent")
        assert result[0]["validation_status"] == "NOT_APPLICABLE", f"Failed for code: {code}"


def test_validate_dcms_no_dcm_files(tmp_path):
    """A directory with no .dcm files must produce NO_DCM_FILES."""
    case_dir = tmp_path / "BP001_folder"
    case_dir.mkdir()
    (case_dir / "info.txt").write_text("not a dicom")

    data_mapping = [{"fixed_name": "BP001", "data_path": "BP001_folder"}]
    result = validate_dcms(data_mapping, str(tmp_path))
    assert result[0]["validation_status"] == "NO_DCM_FILES"


# ---------------------------------------------------------------------------
# _flag_exam_size_outliers
# ---------------------------------------------------------------------------


def test_flag_exam_size_outliers_below_and_above_limit():
    """IQR outliers must be flagged BELOW_LIMIT and ABOVE_LIMIT; inliers stay OK."""
    data = [
        {"fixed_name": "BP001", "validation_status": "OK", "exam_size": 100.0},
        {"fixed_name": "BP002", "validation_status": "OK", "exam_size": 105.0},
        {"fixed_name": "BP003", "validation_status": "OK", "exam_size": 108.0},
        {"fixed_name": "BP004", "validation_status": "OK", "exam_size": 110.0},
        {"fixed_name": "BP005", "validation_status": "OK", "exam_size": 115.0},
        {"fixed_name": "BP006", "validation_status": "OK", "exam_size": 120.0},
        {"fixed_name": "BP007", "validation_status": "OK", "exam_size": 10.0},   # far below Q1 − 1.5×IQR
        {"fixed_name": "BP008", "validation_status": "OK", "exam_size": 500.0},  # far above Q3 + 1.5×IQR
    ]
    result = _flag_exam_size_outliers(data)
    statuses = {d["exam_size"]: d["validation_status"] for d in result}
    assert statuses[10.0] == "BELOW_LIMIT"
    assert statuses[500.0] == "ABOVE_LIMIT"
    assert statuses[100.0] == "OK"
    assert statuses[115.0] == "OK"


def test_flag_exam_size_outliers_no_ok_exams_returns_early():
    """When no OK exams exist the function must return data unchanged."""
    data = [
        {"validation_status": "FAILED",        "exam_size": 100.0},
        {"validation_status": "MISSING_CLASS", "exam_size": 200.0},
    ]
    result = _flag_exam_size_outliers(data)
    assert all(d["validation_status"] not in ("BELOW_LIMIT", "ABOVE_LIMIT") for d in result)
