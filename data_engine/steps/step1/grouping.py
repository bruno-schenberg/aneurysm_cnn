"""Resolution group classification.

Assigns each exam to one of 8 acquisition groups based on matrix size,
XY pixel spacing, and Z slice spacing. Groups match the empirical clusters
found in the dataset (see reports/resolution_groups.csv).
"""

import pandas as pd


def _assign_res_group(row) -> int | None:
    """Assign resolution group 1-8 from spatial metadata."""
    try:
        r = int(row["rows"])
        c = int(row["columns"])
        sx = float(row["voxel_spacing_col"])
        sz = float(row["voxel_spacing_z"])
    except (ValueError, TypeError):
        return None
    if r == 768 and c == 768 and sz < 0.6:
        return 1
    if r == 768 and c == 696 and sz < 0.6:
        return 2
    if r == 512 and c == 512 and sx >= 0.41 and sz >= 0.5:
        return 3
    if r == 512 and c == 512 and sx < 0.41:
        return 4
    if r == 512 and c == 512 and sz < 0.4:
        return 5
    if r == 768 and c == 768 and sz >= 0.7:
        return 6
    if r == 512 and c == 512 and sx >= 0.41 and 0.35 <= sz < 0.5:
        return 7
    if r == 640:
        return 8
    return None


def assign_res_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``res_group`` column (int 1-8 or NaN) to the pipeline DataFrame."""
    df = df.copy()
    df["res_group"] = df.apply(_assign_res_group, axis=1)
    return df
