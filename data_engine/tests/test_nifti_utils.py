import pytest
import os
from src.nifti_utils import filter_for_conversion

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
