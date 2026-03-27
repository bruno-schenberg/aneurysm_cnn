import os
import pytest
from src.file_utils import (
    add_data_codes,
    add_data_paths,
    count_dcm_files,
    find_missing_cases,
    generate_new_names,
    get_subfolders,
)

def test_generate_new_names():
    folder_list = ["bp1", "BP2", "bp03", "case100", "other_folder"]
    # generate_new_names sorts the list: ["BP2", "bp03", "bp1", "case100", "other_folder"]
    # BP2 -> BP002
    # bp03 -> BP003
    # bp1 -> BP001
    # other_folder -> other_folder (no match)
    # case100 -> case100 (no match)
    
    mapping = generate_new_names(folder_list)
    
    # Sort by original name to verify
    mapping_dict = {m['original_name']: m['fixed_name'] for m in mapping}
    assert mapping_dict["bp1"] == "BP001"
    assert mapping_dict["BP2"] == "BP002"
    assert mapping_dict["bp03"] == "BP003"
    assert mapping_dict["case100"] == "case100"

def test_generate_new_names_duplicates():
    folder_list = ["bp1", "bp001"]
    # bp1 -> BP001
    # bp001 -> BP001
    # Sorted: ["bp001", "bp1"]
    # bp001 -> BP001A
    # bp1 -> BP001B
    mapping = generate_new_names(folder_list)
    mapping_dict = {m['original_name']: m['fixed_name'] for m in mapping}
    assert mapping_dict["bp001"] == "BP001A"
    assert mapping_dict["bp1"] == "BP001B"

def test_add_data_codes():
    mapping = [
        {"original_name": "f1", "direct_items": 10, "non_empty_subfolders": []}, # READY
        {"original_name": "f2", "direct_items": 0, "non_empty_subfolders": ["sub1"]}, # SUBFOLDER_PATH
        {"original_name": "f3", "direct_items": 0, "non_empty_subfolders": []}, # EMPTY
        {"original_name": "f4", "direct_items": 5, "non_empty_subfolders": ["sub1"]}, # DUPLICATE_DATA
        {"fixed_name": "BP999"} # MISSING (no direct_items)
    ]
    updated = add_data_codes(mapping)
    codes = {m.get('original_name', m.get('fixed_name')): m['data_code'] for m in updated}
    assert codes["f1"] == "READY"
    assert codes["f2"] == "SUBFOLDER_PATH"
    assert codes["f3"] == "EMPTY"
    assert codes["f4"] == "DUPLICATE_DATA"
    assert codes["BP999"] == "MISSING"

def test_add_data_paths():
    mapping = [
        {"original_name": "f1", "data_code": "READY"},
        {"original_name": "f2", "data_code": "SUBFOLDER_PATH", "non_empty_subfolders": ["sub1"]},
        {"original_name": "f3", "data_code": "EMPTY"}
    ]
    updated = add_data_paths(mapping)
    paths = {m['original_name']: m['data_path'] for m in updated}
    assert paths["f1"] == "f1"
    assert paths["f2"] == os.path.join("f2", "sub1")
    assert paths["f3"] == "EMPTY"


# ---------------------------------------------------------------------------
# get_subfolders
# ---------------------------------------------------------------------------


def test_get_subfolders_returns_subfolder_names(tmp_path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "file.txt").write_text("not a dir")
    result = get_subfolders(str(tmp_path))
    assert set(result) == {"alpha", "beta"}


def test_get_subfolders_missing_directory_returns_empty():
    result = get_subfolders("/nonexistent/path/that/does/not/exist")
    assert result == []


# ---------------------------------------------------------------------------
# count_dcm_files
# ---------------------------------------------------------------------------


def test_count_dcm_files_counts_only_dcm_extensions(tmp_path):
    (tmp_path / "slice001.dcm").write_bytes(b"fake")
    (tmp_path / "slice002.DCM").write_bytes(b"fake")  # uppercase extension
    (tmp_path / "readme.txt").write_text("ignored")
    (tmp_path / "subdir").mkdir()
    assert count_dcm_files(str(tmp_path)) == 2


def test_count_dcm_files_missing_directory_returns_zero():
    assert count_dcm_files("/nonexistent/path/xyz") == 0


# ---------------------------------------------------------------------------
# find_missing_cases
# ---------------------------------------------------------------------------


def test_find_missing_cases_identifies_gaps_in_sequence():
    # Provide cases BP001 and BP003, expect BP002 flagged as missing
    name_mapping = [
        {"original_name": "bp1", "fixed_name": "BP001"},
        {"original_name": "bp3", "fixed_name": "BP003"},
    ]
    missing = find_missing_cases(name_mapping)
    missing_names = {m["fixed_name"] for m in missing}
    assert "BP002" in missing_names
    assert "BP001" not in missing_names
    assert "BP003" not in missing_names
    assert all(m["original_name"] == "missing" for m in missing)


def test_find_missing_cases_empty_mapping_returns_all_1_to_999():
    missing = find_missing_cases([])
    assert len(missing) == 999
    fixed_names = {m["fixed_name"] for m in missing}
    assert "BP001" in fixed_names
    assert "BP999" in fixed_names


# ---------------------------------------------------------------------------
# add_data_codes — ≥2 subfolders branch
# ---------------------------------------------------------------------------


def test_add_data_codes_duplicate_data_multiple_subfolders():
    """Two or more non-empty subfolders must produce DUPLICATE_DATA."""
    mapping = [{"original_name": "f1", "direct_items": 0, "non_empty_subfolders": ["sub1", "sub2"]}]
    updated = add_data_codes(mapping)
    assert updated[0]["data_code"] == "DUPLICATE_DATA"
