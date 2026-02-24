import pytest
import numpy as np
from src.dicom_utils import get_orientation

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
