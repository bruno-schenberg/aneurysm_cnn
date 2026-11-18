"""Shared helpers for the one-off justification/diagnostic scripts in this
directory: NIfTI lookup by exam ID, exam loading, display normalization,
and opening a rendered output in the OS viewer.
"""
import shutil
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
from medprep.visualize import percentile_normalize as _clip_to_percentile

import config


def open_file(path) -> None:
    """Open path in the OS default viewer, if a viewer is available.

    Uses xdg-open (Linux-only). On macOS/Windows or a headless environment
    without xdg-open installed, this prints a note instead of raising —
    the PNG is still saved, just not auto-opened.
    """
    if shutil.which("xdg-open") is None:
        print(f"(auto-open skipped — xdg-open not found; open manually: {path})")
        return
    subprocess.Popen(["xdg-open", str(path)])


def find_nifti(exam_id: str) -> tuple[str | None, str | None]:
    """Return (path, class_label) for exam_id's NIfTI file under
    config.NIFTI_ROOT, or (None, None) if not found."""
    for cls_dir in config.NIFTI_ROOT.iterdir():
        p = cls_dir / f"{exam_id}.nii.gz"
        if p.exists():
            return str(p), cls_dir.name
    return None, None


def load_exam(exam_id: str):
    """Load an exam, return (data, spacing, class_label) or (None, None, None)."""
    path, cls = find_nifti(exam_id)
    if path is None:
        return None, None, None
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    sp = img.header.get_zooms()
    return data, sp, cls


def load_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Group converted exams by (rows, columns, XY voxel spacing) acquisition
    shape. Returns (converted_df, groups_df, key) — groups_df has one row
    per acquisition group with exam count ("n") and median Z extent
    ("fov_z_mm"), sorted largest group first.
    """
    converted = df[df["qc_block_reason"].isna() | (df["qc_block_reason"] == "")].copy()
    converted["sx"] = converted["voxel_spacing_row"].round(4)
    converted["rows_i"] = converted["rows"].astype(int)
    converted["cols_i"] = converted["columns"].astype(int)
    key = ["rows_i", "cols_i", "sx"]
    groups = (
        converted.groupby(key)
        .agg(n=("exam_id", "count"), fov_z_mm=("extent_z_mm", "median"))
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    return converted, groups, key


def norm_percentile(vol: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] of foreground and rescale to [0, 1], background at 0.

    Clipping delegates to medprep.visualize.percentile_normalize; the [0, 1]
    rescale and explicit background pin are display-specific to these scripts
    (medprep's version only clips, and doesn't zero out sub-p1 background).
    """
    fg = vol[vol > 0]
    if len(fg) == 0:
        return vol.copy()
    p1, p99 = np.percentile(fg, [1, 99])
    clipped = _clip_to_percentile(vol)
    out = np.where(vol > 0, (clipped - p1) / max(p99 - p1, 1e-9), 0.0)
    return np.clip(out, 0, None).astype(np.float32)
