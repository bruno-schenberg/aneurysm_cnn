import csv

import pytest

from src.class_utils import check_missing_class, join_class_data


@pytest.fixture
def classes_csv(tmp_path):
    """Write a minimal classes.csv and return its path as a string."""
    csv_path = tmp_path / "classes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exam", "class", "location", "Age"])
        writer.writeheader()
        writer.writerows([
            {"exam": "BP001", "class": "0", "location": "anterior", "Age": "45"},
            {"exam": "BP002", "class": "1", "location": "middle",   "Age": "62"},
        ])
    return str(csv_path)


# ---------------------------------------------------------------------------
# join_class_data
# ---------------------------------------------------------------------------


def test_join_class_data_matches_base_name(classes_csv):
    """Exact match: fixed_name 'BP001' joins class '0' from the CSV."""
    data = [{"fixed_name": "BP001", "validation_status": "OK"}]
    result = join_class_data(data, classes_csv)
    assert result[0]["class"] == "0"
    assert result[0]["location"] == "anterior"


def test_join_class_data_strips_suffix_for_matching(classes_csv):
    """Suffix 'A' in 'BP002A' must be stripped so it matches 'BP002' in the CSV."""
    data = [{"fixed_name": "BP002A", "validation_status": "OK"}]
    result = join_class_data(data, classes_csv)
    assert result[0]["class"] == "1"


def test_join_class_data_file_not_found_returns_data_unchanged():
    """Missing CSV must log a warning and return the original data list intact."""
    data = [{"fixed_name": "BP001", "validation_status": "OK"}]
    result = join_class_data(data, "/nonexistent/classes.csv")
    assert result is data
    assert "class" not in result[0]


def test_join_class_data_no_match_leaves_item_without_class(classes_csv):
    """A fixed_name not present in the CSV must not add a 'class' key."""
    data = [{"fixed_name": "BP999", "validation_status": "OK"}]
    result = join_class_data(data, classes_csv)
    assert result[0].get("class") is None


# ---------------------------------------------------------------------------
# check_missing_class
# ---------------------------------------------------------------------------


def test_check_missing_class_flags_ok_without_class():
    """An OK exam with no class joined must be re-flagged as MISSING_CLASS."""
    data = [{"validation_status": "OK", "fixed_name": "BP001"}]
    result = check_missing_class(data)
    assert result[0]["validation_status"] == "MISSING_CLASS"


def test_check_missing_class_ok_with_class_is_unchanged():
    """An OK exam that already has a class must remain OK."""
    data = [{"validation_status": "OK", "class": "1", "fixed_name": "BP002"}]
    result = check_missing_class(data)
    assert result[0]["validation_status"] == "OK"


def test_check_missing_class_ignores_non_ok_statuses():
    """Exams with a status other than OK must not be re-flagged even if classless."""
    data = [
        {"validation_status": "MIXED_SERIES_ERROR", "fixed_name": "BP003"},
        {"validation_status": "VARIABLE_SPACING",   "fixed_name": "BP004"},
    ]
    result = check_missing_class(data)
    assert result[0]["validation_status"] == "MIXED_SERIES_ERROR"
    assert result[1]["validation_status"] == "VARIABLE_SPACING"
