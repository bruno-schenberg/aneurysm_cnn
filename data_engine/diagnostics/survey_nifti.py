"""
survey_nifti.py

Reads every NIfTI file under /mnt/data/nifti and writes a CSV with
per-exam dimensions, voxel spacing, and real-world physical size.

Output: data_engine/nifti_survey.csv

Usage:
    python data_engine/survey_nifti.py [--nifti-dir /mnt/data/nifti]
"""

import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np


FIELDNAMES = [
    "exam",
    "class",
    "dim_x",
    "dim_y",
    "dim_z",
    "spacing_x_mm",
    "spacing_y_mm",
    "spacing_z_mm",
    "size_x_mm",
    "size_y_mm",
    "size_z_mm",
    "min_spacing_mm",
    "max_spacing_mm",
    "is_isotropic",
]


def survey_file(nifti_path: str, label: str) -> dict:
    img = nib.load(nifti_path)

    # Shape — nibabel returns (x, y, z) for 3-D volumes.
    shape = img.shape[:3]
    dim_x, dim_y, dim_z = int(shape[0]), int(shape[1]), int(shape[2])

    # Voxel spacing in mm from the header zooms (always positive).
    zooms = img.header.get_zooms()
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])

    # Physical field-of-view in mm.
    size_x = dim_x * sx
    size_y = dim_y * sy
    size_z = dim_z * sz

    min_sp = min(sx, sy, sz)
    max_sp = max(sx, sy, sz)
    # Treat as isotropic if all spacings are within 5% of each other.
    is_isotropic = (max_sp - min_sp) / max_sp < 0.05

    return {
        "exam": os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0],
        "class": label,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "dim_z": dim_z,
        "spacing_x_mm": round(sx, 4),
        "spacing_y_mm": round(sy, 4),
        "spacing_z_mm": round(sz, 4),
        "size_x_mm": round(size_x, 1),
        "size_y_mm": round(size_y, 1),
        "size_z_mm": round(size_z, 1),
        "min_spacing_mm": round(min_sp, 4),
        "max_spacing_mm": round(max_sp, 4),
        "is_isotropic": is_isotropic,
    }


def main(nifti_dir: str, output_csv: str) -> None:
    rows = []
    errors = []

    for label in ("0", "1"):
        class_dir = os.path.join(nifti_dir, label)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skipping class {label}.")
            continue

        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".nii.gz"))
        print(f"Class {label}: {len(files)} files")

        for fname in files:
            path = os.path.join(class_dir, fname)
            try:
                rows.append(survey_file(path, label))
            except Exception as e:
                errors.append((fname, str(e)))
                print(f"  ERROR {fname}: {e}", file=sys.stderr)

    rows.sort(key=lambda r: r["exam"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {output_csv}")
    if errors:
        print(f"{len(errors)} files failed — see stderr above.")

    # Quick summary stats.
    if rows:
        import statistics

        def col(name):
            return [r[name] for r in rows]

        print("\n--- Summary ---")
        for axis, d_col, s_col, sz_col in [
            ("X", "dim_x", "spacing_x_mm", "size_x_mm"),
            ("Y", "dim_y", "spacing_y_mm", "size_y_mm"),
            ("Z", "dim_z", "spacing_z_mm", "size_z_mm"),
        ]:
            dims = col(d_col)
            spacings = col(s_col)
            sizes = col(sz_col)
            print(
                f"  {axis}  dim  {min(dims):4d}–{max(dims):4d}  "
                f"spacing {min(spacings):.3f}–{max(spacings):.3f} mm  "
                f"FOV {min(sizes):.0f}–{max(sizes):.0f} mm"
            )

        isotropic = sum(1 for r in rows if r["is_isotropic"])
        print(f"\n  Isotropic (spacing ratio <5%): {isotropic}/{len(rows)}")

        min_sps = col("min_spacing_mm")
        max_sps = col("max_spacing_mm")
        print(f"  In-plane spacing:  {min(min_sps):.3f}–{max(max_sps):.3f} mm")
        print(f"  Median min spacing: {statistics.median(min_sps):.3f} mm")
        print(f"  Median max spacing: {statistics.median(max_sps):.3f} mm")

    # Unique size × spacing configurations with counts.
    seen: dict[tuple, str] = {}
    counts: dict[tuple, int] = {}
    for r in rows:
        key = (r["dim_x"], r["dim_y"], r["dim_z"],
               r["spacing_x_mm"], r["spacing_y_mm"], r["spacing_z_mm"])
        if key not in seen:
            seen[key] = r["exam"]
        counts[key] = counts.get(key, 0) + 1

    variants_csv = os.path.splitext(output_csv)[0] + "_variants.csv"
    with open(variants_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim_x", "dim_y", "dim_z",
                                               "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
                                               "count", "example"])
        writer.writeheader()
        for (dx, dy, dz, sx, sy, sz), example in sorted(seen.items()):
            key = (dx, dy, dz, sx, sy, sz)
            writer.writerow({"dim_x": dx, "dim_y": dy, "dim_z": dz,
                             "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": sz,
                             "count": counts[key], "example": example})

    total = sum(counts.values())
    print(f"\n--- Unique size × spacing configurations: {len(seen)} (total exams: {total}) ---")
    print(f"  {'dim_x':>6} {'dim_y':>6} {'dim_z':>6}  {'sx':>7} {'sy':>7} {'sz':>7}  {'count':>6}  example")
    for (dx, dy, dz, sx, sy, sz), example in sorted(seen.items()):
        print(f"  {dx:>6} {dy:>6} {dz:>6}  {sx:>7} {sy:>7} {sz:>7}  {counts[(dx,dy,dz,sx,sy,sz)]:>6}  {example}")
    print(f"Wrote {len(seen)} rows → {variants_csv}")

    # Group similar configurations.
    # Two configs are in the same group when they share the same XY matrix size
    # AND their XY spacing is within 15% AND their Z spacing is within 20%.
    # Z slice count is intentionally excluded — the same scanner/protocol can
    # produce a different number of slices depending on how much of the head
    # was covered in a given acquisition.
    configs = sorted(seen.keys())  # list of (dx, dy, dz, sx, sy, sz)

    group_id: dict[tuple, int] = {}
    next_id = 1
    for cfg in configs:
        dx, dy, dz, sx, sy, sz = cfg
        assigned = None
        for rep_cfg, gid in group_id.items():
            rdx, rdy, rdz, rsx, rsy, rsz = rep_cfg
            same_xy_matrix = (dx == rdx and dy == rdy)
            sx_close = abs(sx - rsx) / max(sx, rsx) < 0.15
            sz_close = abs(sz - rsz) / max(sz, rsz) < 0.20
            if same_xy_matrix and sx_close and sz_close:
                assigned = gid
                break
        if assigned is None:
            assigned = next_id
            next_id += 1
        group_id[cfg] = assigned

    # Build group rows: one row per config, with group_id and group_count.
    # group_count = total exams across all configs in the same group.
    group_totals: dict[int, int] = {}
    for cfg, gid in group_id.items():
        group_totals[gid] = group_totals.get(gid, 0) + counts[cfg]

    groups_csv = os.path.splitext(output_csv)[0] + "_groups.csv"
    with open(groups_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group_id", "group_count",
            "dim_x", "dim_y", "dim_z",
            "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
            "count", "example",
        ])
        writer.writeheader()
        for cfg in configs:
            dx, dy, dz, sx, sy, sz = cfg
            gid = group_id[cfg]
            writer.writerow({
                "group_id":    gid,
                "group_count": group_totals[gid],
                "dim_x": dx, "dim_y": dy, "dim_z": dz,
                "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": sz,
                "count":   counts[cfg],
                "example": seen[cfg],
            })

    n_groups = len(set(group_id.values()))
    print(f"\n--- Acquisition groups (same XY matrix + sx within 15% + sz within 20%): {n_groups} ---")
    for gid in sorted(set(group_id.values())):
        members = [(cfg, counts[cfg]) for cfg, g in group_id.items() if g == gid]
        members.sort()
        total = group_totals[gid]
        dx, dy, dz, sx, sy, sz = members[0][0]
        print(f"  Group {gid:>2}  ({total:>3} exams)  XY={dx}x{dy}  "
              f"sx={sx:.4f}  sz={sz:.4f}  [{len(members)} config(s)]")
        for cfg, cnt in members:
            cdx, cdy, cdz, csx, csy, csz = cfg
            print(f"             {seen[cfg]:>6}  {cdx}x{cdy}x{cdz}  "
                  f"sx={csx}  sz={csz}  n={cnt}")
    print(f"Wrote {n_groups} groups → {groups_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Survey NIfTI dimensions and spacing.")
    parser.add_argument("--nifti-dir", default="/mnt/data/nifti")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "nifti_survey.csv"),
    )
    args = parser.parse_args()
    main(args.nifti_dir, args.output)
