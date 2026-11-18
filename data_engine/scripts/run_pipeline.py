#!/usr/bin/env python3
"""
run_pipeline.py — data_engine pipeline orchestrator.

Runs all 4 steps in sequence, or a single step via --step.

    python scripts/run_pipeline.py [--dry-run] [--workers N]
    python scripts/run_pipeline.py --step 0
    python scripts/run_pipeline.py --step 1 --dry-run
    python scripts/run_pipeline.py --step 2 --sample --variants iso_128
    python scripts/run_pipeline.py --test

Steps:
    0  Extract archives, discover exam folders, assign IDs.
       Output: reports/step0_discover.csv
    1  Join metadata, run QC, assign resolution groups, convert to NIfTI.
       Output: reports/step1_convert.csv  (+mixed_series_report.csv)
    2  Resample NIfTI files into dataset variants.
       Output: reports/step2_resample.csv
    3  Visual QC — render pipeline stages side-by-side.
       Output: reports/step3_visualize.png
"""
import argparse
import sys
from pathlib import Path

import config


def main() -> None:
    parser = argparse.ArgumentParser(description="data_engine pipeline (steps 0-3)")
    parser.add_argument("--step", type=int, nargs="+", choices=[0, 1, 2, 3], default=None,
                        help="Run these steps (default: all 4 in sequence).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip extraction (step 0) and conversion (step 1).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for conversion/resampling (default: 4).")
    parser.add_argument("--sample-csv", type=Path, default=None,
                        help="Step 1: restrict to exam_ids listed in this CSV.")
    parser.add_argument("--sample", action="store_true",
                        help="Step 2: restrict to sample_exams.csv.")
    parser.add_argument("--variants", nargs="+", default=None, metavar="NAME",
                        help="Step 2: variant names to generate (default: all).")
    parser.add_argument("--test", action="store_true",
                        help="Use sample paths under /mnt/data/ (sample_zips, sample_raw, etc.).")
    args = parser.parse_args()

    if args.test:
        config.enable_test_mode()

    # Resolve variant names to dicts
    # In test mode, variant names are prefixed with "sample_" automatically,
    # so --variants iso_128 matches sample_iso_128.
    selected_variants = None
    if args.variants:
        all_variants = {v["name"]: v for v in config.VARIANTS}
        resolved = []
        unknown = []
        for n in args.variants:
            if n in all_variants:
                resolved.append(all_variants[n])
            elif f"sample_{n}" in all_variants:
                resolved.append(all_variants[f"sample_{n}"])
            else:
                unknown.append(n)
        if unknown:
            print(f"ERROR: unknown variant(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(all_variants)}")
            sys.exit(1)
        selected_variants = resolved

    steps = args.step if args.step is not None else [0, 1, 2, 3]

    if 0 in steps:
        from steps.step0 import run as run_step0
        print("=" * 60)
        print("STEP 0 — Discover")
        print("=" * 60)
        run_step0(dry_run=args.dry_run)
        print()

    if 1 in steps:
        from steps.step1 import run as run_step1
        print("=" * 60)
        print("STEP 1 — Convert")
        print("=" * 60)
        run_step1(
            dry_run=args.dry_run,
            workers=args.workers,
            sample_csv=args.sample_csv,
        )
        print()

    if 2 in steps:
        from steps.step2 import run as run_step2
        print("=" * 60)
        print("STEP 2 — Resample")
        print("=" * 60)
        run_step2(
            variants=selected_variants,
            workers=args.workers,
            sample=args.sample,
        )
        print()

    if 3 in steps:
        from steps.step3 import run as run_step3
        print("=" * 60)
        print("STEP 3 — Visualize")
        print("=" * 60)
        run_step3()
        print()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
