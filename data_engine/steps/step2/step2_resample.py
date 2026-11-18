"""Step 2 — Generate resampled dataset variants from NIfTI files.

Reads NIfTI files produced by step 1 and writes one dataset directory per
variant under datasets/<variant_name>/<class>/<exam_id>.nii.gz.

Output: reports/step2_resample.csv
"""
import sys

import pandas as pd

import config
from medprep.resample import run_resampling


def run(
    variants: list[dict] | None = None,
    workers: int = 4,
    sample: bool = False,
    append: bool = False,
):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Resolve variants ----
    if variants is None:
        variants = list(config.VARIANTS)

    # ---- Resolve sample IDs ----
    sample_ids: set[str] | None = None
    if sample:
        sample_csv = config.DATASET_CSV.parent / "sample_exams.csv"
        if not sample_csv.exists():
            print(f"ERROR: sample CSV not found: {sample_csv}")
            print("Run scripts/make_sample.py first.")
            sys.exit(1)
        sample_ids = set(
            pd.read_csv(sample_csv)["exam_id"].dropna().astype(str)
        )
        print(f"[step2] Sample mode: {len(sample_ids)} exam IDs from {sample_csv}")

    # ---- Validate input ----
    if not config.NIFTI_ROOT.exists():
        print(f"ERROR: nifti root not found: {config.NIFTI_ROOT}")
        print("Run step 1 first.")
        sys.exit(1)

    mode_label = "sample" if sample else "full"
    print(f"[step2] Generating {len(variants)} variant(s) [{mode_label}] "
          f"with {workers} worker(s):")
    for v in variants:
        print(f"  {v['name']:10s}  strategy={v['strategy']:20s}  "
              f"shape={'x'.join(str(s) for s in v['output_shape'])}")

    # ---- Run ----
    df = run_resampling(
        nifti_root=config.NIFTI_ROOT,
        datasets_root=config.DATASETS_ROOT,
        variants=variants,
        sample_ids=sample_ids,
        max_workers=workers,
    )

    if df.empty:
        print("No files processed.")
        return

    # ---- Save ----
    if append and config.STEP2_REPORT.exists():
        existing = pd.read_csv(config.STEP2_REPORT)
        df = pd.concat([existing, df], ignore_index=True)

    _COLUMN_ORDER = [
        "exam_id", "class", "variant", "resample_status",
        "output_shape", "input_shape",
        "actual_spacing_x", "actual_spacing_y", "actual_spacing_z",
        "pct_pad", "pct_crop", "strategy",
    ]
    ordered = [c for c in _COLUMN_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    df.to_csv(config.STEP2_REPORT, index=False)

    counts = df["resample_status"].value_counts()
    for status, n in sorted(counts.items()):
        print(f"  {status}: {int(n)}")
    print(f"\nStep 2 report -> {config.STEP2_REPORT}  ({len(df)} rows)")
