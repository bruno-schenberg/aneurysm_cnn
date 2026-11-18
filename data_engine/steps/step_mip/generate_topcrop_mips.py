"""Generate MIP datasets from original NIfTIs with top-Z cropping.

Instead of the symmetric center crop (c90z), this keeps the TOP N% of Z slices,
removing only the bottom. Aneurysms sit at the Circle of Willis (above center),
so top crop preserves all aneurysm signal while removing neck/jaw noise.

Generates 3-channel MIPs (axial/coronal/sagittal stacked) for each crop variant.

Output datasets (under --output-root), one per --crops value in a single run:
  mip_3ch_top90/  — top 90% of Z (drop bottom 10%)
  mip_3ch_top80/  — top 80% of Z (drop bottom 20%)
  mip_3ch_top70/  — top 70% of Z (drop bottom 30%)
  mip_3ch_top60/  — top 60% of Z (drop bottom 40%, matches Lausanne)

--suffix appends to every variant name (e.g. --suffix _512 with the default
--crops produces mip_3ch_top90_512/, mip_3ch_top80_512/, ...) — use it to
generate a second batch (e.g. a different --target-size) into the same
--output-root without colliding with an existing run's variant dirs.

Usage:
    python -m steps.step_mip.generate_topcrop_mips \
        --source /path/to/nifti \
        --output-root /path/to/datasets \
        --crops 90 80 70 60

    # A second, differently-sized batch alongside the one above:
    python -m steps.step_mip.generate_topcrop_mips \
        --source /path/to/nifti \
        --output-root /path/to/datasets \
        --target-size 512 --suffix _512
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from medprep.mip import PROJECTION_AXES, load_volume, resize_2d, top_z_crop

from ._cli import collect_class_files, ensure_variant_dirs, file_stem, print_variant_counts


def process_volume(
    nifti_path: str | Path, top_pct: int, target_size: int = 384
) -> np.ndarray:
    """Load NIfTI, apply top-Z crop, generate 3-channel MIP [axial, coronal, sagittal]."""
    data = top_z_crop(load_volume(nifti_path), top_pct)

    target = (target_size, target_size)
    mips = []
    for axis in PROJECTION_AXES.values():
        mip = np.max(data, axis=axis)
        mips.append(resize_2d(mip, target))

    return np.stack(mips, axis=0)  # shape: (3, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate top-Z-cropped MIP datasets from original NIfTIs"
    )
    parser.add_argument(
        "--source", required=True,
        help="Source NIfTI dir with 0/ and 1/ subdirs (original resolution)",
    )
    parser.add_argument("--output-root", required=True, help="Parent dir for MIP datasets")
    parser.add_argument(
        "--crops", nargs="+", type=int, default=[90, 80, 70, 60],
        help="Top-Z crop percentages to generate (default: 90 80 70 60)",
    )
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--suffix", default="", help="Suffix for variant name (e.g. '_512')")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output_root)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    variant_names = [f"mip_3ch_top{c}{args.suffix}" for c in args.crops]
    ensure_variant_dirs(output_root, variant_names)

    # Process all volumes
    for cls in ("0", "1"):
        files = [f for f, _ in collect_class_files(source_dir, (".nii", ".nii.gz"), classes=(cls,))]

        for i, fpath in enumerate(files):
            stem = file_stem(fpath)

            for crop_pct, variant in zip(args.crops, variant_names):
                out_path = output_root / variant / cls / f"{stem}.npy"
                if out_path.exists():
                    continue

                mip_3ch = process_volume(fpath, crop_pct, args.target_size)
                np.save(str(out_path), mip_3ch)

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  [{i+1}/{len(files)}] {stem}")

    print(f"\nDone.")
    print_variant_counts(output_root, variant_names)


if __name__ == "__main__":
    main()
