"""Merge MIP datasets from our data and Lausanne into combined directories.

Creates mip_*_mixed/ datasets by copying .npy files from both sources.
Files are prefixed with 'ours_' or 'laus_' to avoid name collisions.

Usage:
    python -m steps.step_mip.merge_datasets --datasets-root /path/to/datasets
"""

import argparse
import shutil
from pathlib import Path


VARIANTS = ("mip_axial", "mip_coronal", "mip_sagittal", "mip_3ch")


def merge_variant(datasets_root: Path, variant: str) -> None:
    ours_dir = datasets_root / variant
    laus_dir = datasets_root / f"{variant}_lausanne"
    mixed_dir = datasets_root / f"{variant}_mixed"

    if not ours_dir.exists():
        print(f"  WARNING: {ours_dir} not found, skipping")
        return
    if not laus_dir.exists():
        print(f"  WARNING: {laus_dir} not found, skipping")
        return

    counts = {"ours": 0, "laus": 0}

    for cls in ("0", "1"):
        out_cls = mixed_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)

        # Copy our files (prefix to avoid collisions)
        for f in sorted((ours_dir / cls).glob("*.npy")):
            dst = out_cls / f"ours_{f.name}"
            if not dst.exists():
                shutil.copy2(f, dst)
            counts["ours"] += 1

        # Copy Lausanne files
        for f in sorted((laus_dir / cls).glob("*.npy")):
            dst = out_cls / f"laus_{f.name}"
            if not dst.exists():
                shutil.copy2(f, dst)
            counts["laus"] += 1

    total = sum(1 for _ in mixed_dir.rglob("*.npy"))
    print(f"  {variant}_mixed: {total} files (ours={counts['ours']}, lausanne={counts['laus']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MIP datasets for mixed training")
    parser.add_argument("--datasets-root", required=True, help="Root dir containing MIP datasets")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)

    print("Merging MIP datasets...")
    for variant in VARIANTS:
        merge_variant(datasets_root, variant)

    print("\nDone.")


if __name__ == "__main__":
    main()
