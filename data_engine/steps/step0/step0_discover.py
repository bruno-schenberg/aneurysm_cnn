"""Step 0 — Extract archives, discover exam folders, assign exam IDs.

This step is project-specific: the naming regex (BP####) and folder layout
are tailored to this dataset. Other researchers would replace this step with
their own discovery/naming logic, keeping the output CSV contract for step 1.

Output: reports/step0_discover.csv
"""
import sys

import config
from medprep.discovery import extract_archives, discover_exams
from .naming import assign_exam_ids


def run(dry_run: bool = False):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Extract ----
    if dry_run:
        print("[step0] Extraction skipped (--dry-run).")
    else:
        print(f"[step0] Extracting archives from {config.SOURCE_DIR} ...")
        if not config.SOURCE_DIR.exists():
            print(f"  ERROR: source directory not found: {config.SOURCE_DIR}")
            sys.exit(1)
        extract_archives(config.SOURCE_DIR, config.STAGING_ROOT)

    # ---- Discover ----
    print(f"[step0] Discovering exam folders in {config.STAGING_ROOT} ...")
    if not config.STAGING_ROOT.exists():
        print(f"  ERROR: staging root not found: {config.STAGING_ROOT}")
        sys.exit(1)

    df = discover_exams(config.STAGING_ROOT)

    if df.empty:
        print("  No exam folders found.")
    else:
        for status, count in sorted(df["folder_status"].value_counts().items()):
            print(f"  {status}: {count}")

    # ---- Naming ----
    print("[step0] Assigning exam IDs ...")
    df = assign_exam_ids(df)

    for status, count in sorted(df["name_parse_status"].value_counts().items()):
        print(f"  {status}: {count}")

    # ---- Save ----
    df.to_csv(config.STEP0_REPORT, index=False)
    print(f"\nStep 0 report -> {config.STEP0_REPORT}  ({len(df)} rows)")
