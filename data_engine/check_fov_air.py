"""
check_fov_air.py

For each NIfTI exam in --nifti-dir, measures how deep the air border is on
each of the 6 faces of the volume (x_low, x_high, y_low, y_high, z_low,
z_high).

A voxel slice is classified as air when its maximum value is below
--threshold (default -200 HU). The scan proceeds inward from each face
until the first non-air slice is reached; the depth is recorded in both
voxels and millimetres.

The "content width" per axis is:
    fov_mm  -  air_low_mm  -  air_high_mm

This lets you check whether a 240 mm FOV exam is really just a 180 mm brain
surrounded by 30 mm of air on each XY side.

Assigns each exam a scanner group from nifti_survey_groups.csv using the
same grouping criteria as survey_nifti.py:
    same XY matrix + spacing_x within 15% + spacing_z within 20%

Requires nifti_survey.csv (for per-exam spacing/FOV) and
nifti_survey_groups.csv (for group assignments).

Usage:
    python data_engine/check_fov_air.py
    python data_engine/check_fov_air.py \\
        --nifti-dir        /mnt/data/nifti \\
        --survey-csv       data_engine/nifti_survey.csv \\
        --groups-csv       data_engine/nifti_survey_groups.csv \\
        --output           data_engine/fov_air.csv \\
        --threshold        -200
"""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


FIELDNAMES = [
    "exam", "class", "group_id", "group_count",
    "dim_x", "dim_y", "dim_z",
    "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
    "fov_x_mm", "fov_y_mm", "fov_z_mm",
    # air depth per face
    "air_x_low_px",  "air_x_low_mm",
    "air_x_high_px", "air_x_high_mm",
    "air_y_low_px",  "air_y_low_mm",
    "air_y_high_px", "air_y_high_mm",
    "air_z_low_px",  "air_z_low_mm",
    "air_z_high_px", "air_z_high_mm",
    # derived: actual anatomy extent
    "content_x_mm", "content_y_mm", "content_z_mm",
]


def air_depth(data: np.ndarray, axis: int, from_high: bool, threshold: float) -> int:
    """Return the number of all-air slices from one face along *axis*.

    Scans inward (from index 0 or from the last index) until the first slice
    whose maximum value exceeds *threshold*.  Returns the slice count, which
    equals the depth of the air border in voxels.

    Args:
        data: 3-D float array (HU values).
        axis: 0 = X, 1 = Y, 2 = Z.
        from_high: if True, scan from the high end; otherwise from index 0.
        threshold: slices with max ≤ this value are treated as air.

    Returns:
        Number of air slices (0 if the first slice already has signal).
    """
    n = data.shape[axis]
    indices = range(n - 1, -1, -1) if from_high else range(n)
    for depth, idx in enumerate(indices):
        sl = np.take(data, idx, axis=axis)
        if sl.max() > threshold:
            return depth
    return n  # entire volume is air


def assign_group(exam_row: dict, group_rows: list[dict]) -> dict | None:
    """Find the group that matches this exam's dimensions and spacing.

    Matching criteria (same as survey_nifti.py):
        same XY matrix (dim_x, dim_y)
        spacing_x within 15 %
        spacing_z within 20 %

    Returns the first matching group row, or None if no match.
    """
    dx = int(exam_row["dim_x"])
    dy = int(exam_row["dim_y"])
    sx = float(exam_row["spacing_x_mm"])
    sz = float(exam_row["spacing_z_mm"])

    for g in group_rows:
        if int(g["dim_x"]) != dx or int(g["dim_y"]) != dy:
            continue
        gsx = float(g["spacing_x_mm"])
        gsz = float(g["spacing_z_mm"])
        if abs(sx - gsx) / max(sx, gsx) >= 0.15:
            continue
        if abs(sz - gsz) / max(sz, gsz) >= 0.20:
            continue
        return g
    return None


def process_exam(
    nifti_path: Path,
    survey_row: dict,
    group_rows: list[dict],
    threshold: float,
) -> dict:
    img  = nib.load(nifti_path, mmap=False)
    data = img.get_fdata(dtype=np.float32)

    zooms = img.header.get_zooms()[:3]
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    dx, dy, dz = data.shape[:3]

    group = assign_group(survey_row, group_rows)

    # Air depth (voxels) on each face
    ax_lo = air_depth(data, axis=0, from_high=False, threshold=threshold)
    ax_hi = air_depth(data, axis=0, from_high=True,  threshold=threshold)
    ay_lo = air_depth(data, axis=1, from_high=False, threshold=threshold)
    ay_hi = air_depth(data, axis=1, from_high=True,  threshold=threshold)
    az_lo = air_depth(data, axis=2, from_high=False, threshold=threshold)
    az_hi = air_depth(data, axis=2, from_high=True,  threshold=threshold)

    del data
    gc.collect()

    fov_x = round(dx * sx, 1)
    fov_y = round(dy * sy, 1)
    fov_z = round(dz * sz, 1)

    content_x = round(fov_x - ax_lo * sx - ax_hi * sx, 1)
    content_y = round(fov_y - ay_lo * sy - ay_hi * sy, 1)
    content_z = round(fov_z - az_lo * sz - az_hi * sz, 1)

    return {
        "exam":        survey_row["exam"],
        "class":       survey_row["class"],
        "group_id":    group["group_id"]    if group else "",
        "group_count": group["group_count"] if group else "",
        "dim_x": dx, "dim_y": dy, "dim_z": dz,
        "spacing_x_mm": round(sx, 4),
        "spacing_y_mm": round(sy, 4),
        "spacing_z_mm": round(sz, 4),
        "fov_x_mm": fov_x, "fov_y_mm": fov_y, "fov_z_mm": fov_z,
        "air_x_low_px":  ax_lo, "air_x_low_mm":  round(ax_lo * sx, 1),
        "air_x_high_px": ax_hi, "air_x_high_mm": round(ax_hi * sx, 1),
        "air_y_low_px":  ay_lo, "air_y_low_mm":  round(ay_lo * sy, 1),
        "air_y_high_px": ay_hi, "air_y_high_mm": round(ay_hi * sy, 1),
        "air_z_low_px":  az_lo, "air_z_low_mm":  round(az_lo * sz, 1),
        "air_z_high_px": az_hi, "air_z_high_mm": round(az_hi * sz, 1),
        "content_x_mm": content_x,
        "content_y_mm": content_y,
        "content_z_mm": content_z,
    }


def main(nifti_dir: str, survey_csv: str, groups_csv: str,
         output_csv: str, threshold: float) -> None:

    survey = {r["exam"]: r for r in csv.DictReader(open(survey_csv))}
    group_rows = list(csv.DictReader(open(groups_csv)))

    nifti_files = sorted(Path(nifti_dir).rglob("*/*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found under {nifti_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(nifti_files)} exams | threshold = {threshold} HU")

    rows = []
    errors = []
    for i, path in enumerate(nifti_files, 1):
        exam = path.stem.replace(".nii", "")
        print(f"  [{i}/{len(nifti_files)}] {exam}...", end="\r", flush=True)

        if exam not in survey:
            print(f"\n  WARNING: {exam} not found in survey CSV, skipping.")
            continue
        try:
            row = process_exam(path, survey[exam], group_rows, threshold)
            rows.append(row)
        except Exception as e:
            errors.append((exam, str(e)))
            print(f"\n  ERROR {exam}: {e}", file=sys.stderr)

    rows.sort(key=lambda r: r["exam"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {output_csv}")
    if errors:
        print(f"{len(errors)} exams failed.")

    # Summary by group
    from collections import defaultdict
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)

    print(f"\n{'Group':>7}  {'n':>4}  {'FOV XY':>10}  "
          f"{'med air X low':>14}  {'med air X hi':>13}  "
          f"{'med content X':>14}  {'med content Y':>14}  {'med content Z':>14}")
    print("-" * 100)
    for gid in sorted(by_group):
        g = by_group[gid]
        def med(key): return round(float(np.median([r[key] for r in g])), 1)
        fov_xy = f"{g[0]['fov_x_mm']}x{g[0]['fov_y_mm']}"
        print(f"{gid:>7}  {len(g):>4}  {fov_xy:>10}  "
              f"{med('air_x_low_mm'):>14.1f}  {med('air_x_high_mm'):>13.1f}  "
              f"{med('content_x_mm'):>14.1f}  {med('content_y_mm'):>14.1f}  "
              f"{med('content_z_mm'):>14.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure air border depth on each face of every NIfTI exam."
    )
    parser.add_argument("--nifti-dir",   default="/mnt/data/nifti")
    parser.add_argument("--survey-csv",  default="data_engine/nifti_survey.csv")
    parser.add_argument("--groups-csv",  default="data_engine/nifti_survey_groups.csv")
    parser.add_argument("--output",      default="data_engine/fov_air.csv")
    parser.add_argument(
        "--threshold", type=float, default=-200,
        help="Max HU for a slice to be classified as air (default: -200).",
    )
    args = parser.parse_args()
    main(args.nifti_dir, args.survey_csv, args.groups_csv,
         args.output, args.threshold)
