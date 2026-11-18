"""Step 1 — Metadata join, QC, resolution grouping, DICOM-to-NIfTI conversion.

This step is reusable: given a step 0 CSV with exam_id + dcm_path columns, it
runs scanner-agnostic DICOM quality checks and converts eligible exams to NIfTI.

Output: reports/step1_convert.csv  (+mixed_series_report.csv)
"""
import sys
from pathlib import Path

import pandas as pd

import config
from .metadata import join_metadata
from .qc import run_qc
from .converter import run_conversion
from .grouping import assign_res_groups


_COLUMN_ORDER = [
    "exam_id", "class", "res_group", "brand", "scanner", "field_strength_T",
    "location", "gender", "age",
    "conversion_status", "conversion_error", "qc_block_reason",
    "crop_x0", "crop_y0",
    "folder_status", "name_parse_status",
    "corrupt_headers", "missing_spatial_metadata", "slice_issue",
    "inconsistent_spacing", "extent_issue", "no_valid_series",
    "scout_slices", "mixed_series", "z_order_differs",
    "selected_series_uid",
    "slice_count", "scout_slice_count", "modality", "orientation",
    "rows", "columns", "voxel_spacing_row", "voxel_spacing_col",
    "voxel_spacing_z", "extent_z_mm",
    "series_number", "series_description", "protocol_name",
    "mra_acquisition_type", "scanning_sequence", "scan_options",
    "original_name", "zip_source", "dcm_path", "subfolder_count",
]


def run(
    dry_run: bool = False,
    workers: int = 4,
    sample_csv: Path | None = None,
):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load step 0 output ----
    if not config.STEP0_REPORT.exists():
        print(f"  ERROR: step 0 report not found: {config.STEP0_REPORT}")
        print("  Run step 0 first.")
        sys.exit(1)

    df = pd.read_csv(config.STEP0_REPORT)
    print(f"[step1] Loaded {len(df)} rows from {config.STEP0_REPORT}")

    # ---- Sample filter (optional) ----
    if sample_csv is not None:
        if not sample_csv.exists():
            print(f"  ERROR: sample CSV not found: {sample_csv}")
            sys.exit(1)
        sample_ids = set(pd.read_csv(sample_csv)["exam_id"].dropna().astype(str))
        before = len(df)
        df = df[df["exam_id"].isin(sample_ids)].copy()
        print(f"  [sample] restricted to {len(sample_ids)} exam IDs "
              f"({before} -> {len(df)} rows)")

    # ---- Metadata ----
    print("[step1] Joining metadata from dataset.csv ...")
    df = join_metadata(df, config.DATASET_CSV)

    no_match = df["class"].isna().sum()
    if no_match:
        print(f"  incomplete metadata: {no_match}")

    # ---- QC ----
    print("[step1] Running QC checks ...")
    df, mixed_df = run_qc(df)

    if not mixed_df.empty:
        mixed_df.to_csv(config.MIXED_SERIES_REPORT, index=False)
        print(f"  Mixed series report -> {config.MIXED_SERIES_REPORT}  ({len(mixed_df)} series rows)")

    class_excluded = df[df["class"].isin(config.EXCLUDE_CLASSES)]
    qc_blocked_df  = df[df["qc_block_reason"].notna() & ~df["class"].isin(config.EXCLUDE_CLASSES)]

    if not qc_blocked_df.empty:
        print(f"  qc blocked: {len(qc_blocked_df)}")
        reason_counts = (
            qc_blocked_df["qc_block_reason"].dropna()
            .str.split(",").explode().value_counts()
        )
        for reason, count in reason_counts.items():
            print(f"    {reason}: {int(count)}")

    blocked = df[df["qc_block_reason"].notna()]
    dup_only = blocked[
        blocked["qc_block_reason"].str.split(",").apply(
            lambda parts: set(parts) <= {"slice_issue"})
        & blocked["slice_issue"].astype(str).str.contains("duplicates", na=False)
    ]
    if len(dup_only):
        print(f"  blocked by duplicate slices only: {len(dup_only)}")

    mixed = df["mixed_series"].sum()
    if mixed:
        print(f"  mixed_series (non-blocking): {int(mixed)}")

    z_diff = df["z_order_differs"].sum()
    if z_diff:
        print(f"  z_order_differs (projection != raw Z): {int(z_diff)}")

    eligible_count = (
        (df["folder_status"] == "ready") &
        (~df["name_parse_status"].isin(["unparseable", "gap"])) &
        (~df["class"].isin(config.EXCLUDE_CLASSES))
    ).sum()
    print(f"  exams checked: {eligible_count}")

    if not class_excluded.empty:
        print(f"  excluded by class: {len(class_excluded)}")
        for cls_val, grp in class_excluded.groupby("class"):
            print(f"    class {int(cls_val)}: {len(grp)}")

    # ---- Resolution grouping ----
    df = assign_res_groups(df)
    grp_counts = df["res_group"].dropna().value_counts().sort_index()
    print(f"  res_groups: { {int(g): int(n) for g, n in grp_counts.items()} }")
    unclassified = df["res_group"].isna() & df["qc_block_reason"].isna()
    if unclassified.any():
        print(f"  unclassified (no group match): {unclassified.sum()}")

    # ---- Convert ----
    if dry_run:
        print("[step1] Conversion skipped (--dry-run).")
    else:
        print(f"[step1] Converting eligible exams to NIfTI ({workers} workers) ...")
        print(f"        Output -> {config.NIFTI_ROOT}")
        try:
            df = run_conversion(df, max_workers=workers, nifti_root=config.NIFTI_ROOT)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            df.to_csv(config.STEP1_REPORT, index=False)
            sys.exit(1)

        status_counts = df["conversion_status"].value_counts()
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {int(count)}")

    # ---- Column ordering + save ----
    ordered = [c for c in _COLUMN_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    df.to_csv(config.STEP1_REPORT, index=False)
    print(f"\nStep 1 report -> {config.STEP1_REPORT}  ({len(df)} rows)")
