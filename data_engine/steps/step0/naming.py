"""Stage 3 — Parse folder names into normalised exam IDs.

This step is project-specific: the naming regex (BP####) is tailored to
this dataset. Other researchers would swap in their own parse/format
functions — see medprep.ids.assign_sequential_ids for the generic,
gap-filling ID assignment engine this wraps.
"""
import re

import pandas as pd
from medprep.ids import assign_sequential_ids

_BP_RE = re.compile(r'^[Bb][Pp]\s*(\d+)')


def _parse_id(name: str) -> int | None:
    m = _BP_RE.match(name.strip())
    return int(m.group(1)) if m else None


def _format_id(n: int) -> str:
    return f"BP{n:04d}"


def assign_exam_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add exam_id and name_parse_status columns to df, then append synthetic
    gap rows for every BP number absent from the parsed sequence.

    Duplicates (same numeric ID) are assigned alphabetical suffixes in the
    order they appear in df (discovery already sorts alphabetically).

    name_parse_status values:
      parsed        — unique, clean parse
      parsed_dedup  — duplicate numeric ID, suffix assigned
      unparseable   — regex found no match
      gap           — BP number absent from disk; all other columns are null
    """
    return assign_sequential_ids(
        df,
        parse_id=_parse_id,
        format_id=_format_id,
        name_column="original_name",
        id_column="exam_id",
        status_column="name_parse_status",
    )
