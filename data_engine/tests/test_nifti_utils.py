import pytest
import os
import shutil
import tempfile
import nibabel as nib
from src.nifti_utils import filter_for_conversion, process_and_convert_exams, convert_series_to_nifti

# Path to the sample raw dataset
SAMPLE_RAW_DIR = "/home/aneurysm_cnn/data_engine/dataset/sample-raw"
SAMPLE_DICOM_DIR = os.path.join(SAMPLE_RAW_DIR, "BP05- 2838474", "Anonymous_Mr_712482402_20170105")

def test_filter_for_conversion():
    exam_data = [
        {"fixed_name": "BP001", "validation_status": "OK", "class": "0"},
        {"fixed_name": "BP002", "validation_status": "OK", "class": "1"},
        {"fixed_name": "BP003", "validation_status": "MIXED_SERIES_ERROR", "class": "0"},
        {"fixed_name": "BP004", "validation_status": "OK", "class": "N/A"},
    ]
    eligible = filter_for_conversion(exam_data)
    names = [e['fixed_name'] for e in eligible]
    assert "BP001" in names
    assert "BP002" in names
    assert "BP003" not in names
    assert "BP004" not in names
    assert len(eligible) == 2

def test_convert_series_to_nifti_with_sample_data():
    if not os.path.exists(SAMPLE_DICOM_DIR):
        pytest.skip(f"Sample data not found at {SAMPLE_DICOM_DIR}")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_nifti_path = os.path.join(temp_dir, "test_output.nii.gz")
        
        # Test conversion
        success = convert_series_to_nifti(SAMPLE_DICOM_DIR, output_nifti_path)
        assert success is True, "Conversion should return True for success"
        
        # Verify output exists
        assert os.path.exists(output_nifti_path), "NIfTI output file should be created"
        
        # Verify it's a valid NIfTI file by loading it
        try:
            img = nib.load(output_nifti_path)
            # Basic validation
            assert img.shape is not None, "Loaded image should have a shape"
            assert len(img.shape) >= 3, "Loaded image should be at least 3D"
        except Exception as e:
            pytest.fail(f"Failed to load generated NIfTI file: {e}")

def test_process_and_convert_exams_with_sample_data():
    if not os.path.exists(SAMPLE_DICOM_DIR):
        pytest.skip(f"Sample data not found at {SAMPLE_DICOM_DIR}")

    # Create mock exam data
    mock_exam_name = "Test_Sample_1"
    mock_class = "0"
    exam_data = [{
        "fixed_name": mock_exam_name,
        "data_path": "BP05- 2838474/Anonymous_Mr_712482402_20170105",
        "validation_status": "OK",
        "class": mock_class
    }]
    
    with tempfile.TemporaryDirectory() as temp_out_dir:
        # Run process
        results = process_and_convert_exams(exam_data, SAMPLE_RAW_DIR, temp_out_dir)
        
        # Check result status
        assert len(results) == 1
        assert results[0]["exam_name"] == mock_exam_name
        assert results[0]["status"] == "success"
        
        # Check that file was created in correct subdirectory
        expected_output_path = os.path.join(temp_out_dir, mock_class, f"{mock_exam_name}.nii.gz")
        assert results[0]["output_path"] == expected_output_path
        assert os.path.exists(expected_output_path), "NIfTI file should be created in correct class subfolder"
