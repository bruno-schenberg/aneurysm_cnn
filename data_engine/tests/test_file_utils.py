import pytest
import os
from src.file_utils import generate_new_names, add_data_codes, add_data_paths

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
