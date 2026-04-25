"""
check_resize_quality.py

Measures the real quality of every resize variant by comparing each output
NIfTI against its source exam.  Reads representative exams from
nifti_survey_variants.csv; only exams present under --raw-nifti-dir are
processed, so the script works against both the full dataset and the
12-case sample.

Datasets checked: raw NIfTI + all 8 processed variants (A/B/C/D × 128/256).

Metrics per exam per dataset:
  pad_pct    — % of the output grid that is zero-padding, computed
               geometrically: fraction of output coverage that extends
               beyond the original physical FOV.  0 = no padding added.
  crop_pct   — % of the source physical FOV that was discarded by cropping
               = max(0, (source_fov - output_fov) / source_fov) per axis,
                 then volumetric product.  0 = nothing cropped.
  spacing_mm — effective per-axis voxel spacing in the output

Output: data_engine/resize_quality.csv  +  console summary

Usage:
    python data_engine/check_resize_quality.py
    python data_engine/check_resize_quality.py \\
        --variants-csv  data_engine/nifti_survey_variants.csv \\
        --survey-csv    data_engine/nifti_survey.csv \\
        --raw-nifti-dir /mnt/data/nifti-sample \\
        --datasets-dir  /mnt/data/nifti-sample-datasets \\
        --output        data_engine/resize_quality.csv
"""

import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np


DATASETS = [
    ("raw",   None),                           # raw NIfTI — uses --raw-nifti-dir
    ("A_128", "dataset_A_resampled_cropped"),
    ("B_128", "dataset_B_resampled_shrunk"),
    ("C_128", "dataset_C_cropped"),
    ("D_128", "dataset_D_shrunk"),
    # New 192×192×128 variants
    ("A_192", "dataset_A192"),
    ("B_192", "dataset_B192"),
    ("C_192", "dataset_C192"),
    ("D_192", "dataset_D192"),
    ("E_192", "dataset_E192"),
    ("F_192", "dataset_F192"),
    # New 256×256×176 variants
    ("A_176", "dataset_A176"),
    ("B_176", "dataset_B176"),
    ("C_176", "dataset_C176"),
    ("D_176", "dataset_D176"),
    ("E_176", "dataset_E176"),
    ("F_176", "dataset_F176"),
]

FIELDNAMES = [
    "variant", "size", "exam",
    "orig_dim_x", "orig_dim_y", "orig_dim_z",
    "orig_sp_x", "orig_sp_y", "orig_sp_z",
    "orig_fov_x", "orig_fov_y", "orig_fov_z",
    "shape", "sp_x", "sp_y", "sp_z", "pad_pct", "crop_pct",
]


def crop_pct(orig_fov: tuple, out_shape: tuple, out_spacing: tuple) -> float:
    """
    Fraction of the original physical FOV that was discarded.

    For each axis: output physical coverage = out_shape[i] * out_spacing[i].
    If that is less than the original FOV on that axis, the difference was
    cropped.  The per-axis crop fractions are combined volumetrically.

    Returns a value in [0, 1].  0 means nothing was cropped.
    """
    kept_fracs = []
    for fov, n, sp in zip(orig_fov, out_shape, out_spacing):
        covered = n * sp
        kept = min(covered, fov) / fov if fov > 0 else 1.0
        kept_fracs.append(kept)
    kept_vol = kept_fracs[0] * kept_fracs[1] * kept_fracs[2]
    return round((1.0 - kept_vol) * 100, 1)


def pad_pct(orig_fov: tuple, out_shape: tuple, out_spacing: tuple) -> float:
    """
    Fraction of the output grid that is zero-padding (output extends beyond original FOV).

    Geometric dual of crop_pct.  For each axis, padding occurs when the output
    physical coverage (out_shape[i] * out_spacing[i]) exceeds the original FOV —
    the excess voxels must be zero-padded.  The per-axis padding fractions are
    combined volumetrically.

    Returns a value in [0, 1].  0 means no padding was added by the pipeline.

    This is strictly preferable to counting zero voxels in the output array,
    which conflates genuine pipeline padding with background air voxels that
    were already zero in the source scan.  Variants C and D never add padding
    to clinical volumes (the input is always larger than the target grid), so
    zero-voxel counting would spuriously report the scan's background as
    "padding" for those datasets.
    """
    kept_fracs = []
    for fov, n, sp in zip(orig_fov, out_shape, out_spacing):
        covered = n * sp
        kept = min(covered, fov) / covered if covered > 0 else 1.0
        kept_fracs.append(kept)
    kept_vol = kept_fracs[0] * kept_fracs[1] * kept_fracs[2]
    return round((1.0 - kept_vol) * 100, 1)


def analyse_exam(exam, cls, source_row, datasets_dir, raw_nifti_dir):
    orig_fov = (
        float(source_row["size_x_mm"]),
        float(source_row["size_y_mm"]),
        float(source_row["size_z_mm"]),
    )

    row = {
        "exam": exam,
        "orig_dim_x": source_row["dim_x"],
        "orig_dim_y": source_row["dim_y"],
        "orig_dim_z": source_row["dim_z"],
        "orig_sp_x":  source_row["spacing_x_mm"],
        "orig_sp_y":  source_row["spacing_y_mm"],
        "orig_sp_z":  source_row["spacing_z_mm"],
        "orig_fov_x": round(orig_fov[0], 1),
        "orig_fov_y": round(orig_fov[1], 1),
        "orig_fov_z": round(orig_fov[2], 1),
    }

    for ds_key, ds_folder in DATASETS:
        if ds_folder is None:
            path = os.path.join(raw_nifti_dir, cls, f"{exam}.nii.gz")
        else:
            path = os.path.join(datasets_dir, ds_folder, cls, f"{exam}.nii.gz")

        try:
            img = nib.load(path)
        except FileNotFoundError:
            for metric in ("shape", "sp_x", "sp_y", "sp_z", "pad_pct", "crop_pct"):
                row[f"{ds_key}_{metric}"] = ""
            continue

        shape = img.shape[:3]
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

        pad  = pad_pct(orig_fov, shape, zooms)
        crop = crop_pct(orig_fov, shape, zooms)

        row[f"{ds_key}_shape"]    = f"{shape[0]}x{shape[1]}x{shape[2]}"
        row[f"{ds_key}_sp_x"]     = round(zooms[0], 4)
        row[f"{ds_key}_sp_y"]     = round(zooms[1], 4)
        row[f"{ds_key}_sp_z"]     = round(zooms[2], 4)
        row[f"{ds_key}_pad_pct"]  = pad
        row[f"{ds_key}_crop_pct"] = crop

    return row


def print_summary(rows):
    print(f"\n{'exam':<8} {'orig_fov_mm':>20}  ", end="")
    for ds_key, _ in DATASETS:
        print(f"  {ds_key:>6}(pad/crop)", end="")
    print()
    print("-" * (30 + len(DATASETS) * 20))

    for r in rows:
        fov = f"{r['orig_fov_x']}x{r['orig_fov_y']}x{r['orig_fov_z']}"
        print(f"{r['exam']:<8} {fov:>20}  ", end="")
        for ds_key, _ in DATASETS:
            pad  = r[f"{ds_key}_pad_pct"]
            crop = r[f"{ds_key}_crop_pct"]
            print(f"  {pad:>5.1f}/{crop:<5.1f}", end="")
        print()

    # Per-dataset aggregate
    print(f"\n{'':30}", end="")
    for ds_key, _ in DATASETS:
        print(f"  {ds_key:>11}", end="")
    print()

    for stat_label, stat_fn in [
        ("median pad%",  lambda vals: round(float(np.median(vals)), 1)),
        ("max pad%",     lambda vals: round(float(np.max(vals)), 1)),
        ("median crop%", lambda vals: round(float(np.median(vals)), 1)),
        ("max crop%",    lambda vals: round(float(np.max(vals)), 1)),
    ]:
        metric = "pad_pct" if "pad" in stat_label else "crop_pct"
        print(f"  {stat_label:<28}", end="")
        for ds_key, _ in DATASETS:
            vals = [r[f"{ds_key}_{metric}"] for r in rows if r.get(f"{ds_key}_{metric}") != ""]
            if vals:
                print(f"  {stat_fn(vals):>11}", end="")
            else:
                print(f"  {'N/A':>11}", end="")
        print()


def main(variants_csv, survey_csv, datasets_dir, raw_nifti_dir, output_csv):
    variants = list(csv.DictReader(open(variants_csv)))
    survey   = {r["exam"]: r for r in csv.DictReader(open(survey_csv))}

    # Only process exams whose raw NIfTI exists under raw_nifti_dir.
    # This lets the script run against the 4-exam sample as well as the full dataset.
    available = [
        v for v in variants
        if os.path.exists(
            os.path.join(raw_nifti_dir, survey[v["example"]]["class"], f"{v['example']}.nii.gz")
        )
    ]
    if not available:
        print(f"No matching exams found under {raw_nifti_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for v in available:
        exam = v["example"]
        cls  = survey[exam]["class"]
        print(f"  {exam}...", end="\r", flush=True)
        rows.append(analyse_exam(exam, cls, survey[exam], datasets_dir, raw_nifti_dir))

    rows.sort(key=lambda r: r["exam"])

    # Reshape from one wide row per exam to one row per (dataset, exam).
    long_rows = []
    for r in rows:
        orig = {k: r[k] for k in (
            "exam",
            "orig_dim_x", "orig_dim_y", "orig_dim_z",
            "orig_sp_x",  "orig_sp_y",  "orig_sp_z",
            "orig_fov_x", "orig_fov_y", "orig_fov_z",
        )}
        for ds_key, _ in DATASETS:
            variant, size = ds_key.split("_") if "_" in ds_key else (ds_key, "-")
            long_rows.append({
                "variant":   variant,
                "size":      size,
                **orig,
                "shape":     r.get(f"{ds_key}_shape",    ""),
                "sp_x":      r.get(f"{ds_key}_sp_x",     ""),
                "sp_y":      r.get(f"{ds_key}_sp_y",     ""),
                "sp_z":      r.get(f"{ds_key}_sp_z",     ""),
                "pad_pct":   r.get(f"{ds_key}_pad_pct",  ""),
                "crop_pct":  r.get(f"{ds_key}_crop_pct", ""),
            })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(long_rows)

    print(f"Wrote {len(long_rows)} rows → {output_csv}      ")
    print_summary(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize quality check for all dataset variants.")
    parser.add_argument("--variants-csv",  default="data_engine/nifti_survey_variants.csv")
    parser.add_argument("--survey-csv",    default="data_engine/nifti_survey.csv")
    parser.add_argument("--raw-nifti-dir", default="/mnt/data/nifti-sample")
    parser.add_argument("--datasets-dir",  default="/mnt/data/nifti-sample-datasets")
    parser.add_argument("--output",        default="data_engine/resize_quality.csv")
    args = parser.parse_args()
    main(args.variants_csv, args.survey_csv, args.datasets_dir, args.raw_nifti_dir, args.output)
