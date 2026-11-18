"""Stage 4 — Join dataset.csv, add class/gender/age columns and missing-data flags.

BP-specific ID handling wraps medprep.metadata.join_metadata's generic,
column-configurable join.
"""
from pathlib import Path

import pandas as pd
from medprep.metadata import join_metadata as _join_metadata


def _strip_suffix(exam_id: str) -> str:
    """BP0250A -> BP0250, BP0034 -> BP0034."""
    return exam_id[:6] if len(exam_id) == 7 and exam_id[6].isalpha() else exam_id


def _normalise_exam_id(raw: str) -> str | None:
    """BP001 -> BP0001, BP999 -> BP0999, etc."""
    try:
        return f"BP{int(str(raw).strip()[2:]):04d}"
    except (ValueError, IndexError):
        return None


def join_metadata(df: pd.DataFrame, classes_csv: Path) -> pd.DataFrame:
    """
    Left-join dataset.csv onto df on exam_id.

    Adds columns:
        class, location, gender, age
    """
    return _join_metadata(
        df,
        classes_csv,
        id_column="exam_id",
        metadata_id_column="exam",
        normalize_metadata_id=_normalise_exam_id,
        strip_id_suffix=_strip_suffix,
        rename_columns={"Age": "age"},
        keep_columns=["class", "location", "gender", "age",
                      "brand", "scanner", "field_strength_T"],
    )
