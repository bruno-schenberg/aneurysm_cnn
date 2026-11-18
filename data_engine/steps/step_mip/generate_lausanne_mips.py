"""Generate MIPs from Lausanne TOF-MRA volumes (raw NIfTIs, variable resolution).

Unlike our c90z_384 data (uniform 384x384x192), Lausanne volumes have variable
sizes and resolutions. This script:
  1. Loads each raw NIfTI
  2. Crops to the top 60% of Z slices (approximate CoW region, like our c90z crop)
  3. Generates MIP projections
  4. Resizes all MIPs to target_size x target_size

Output goes to mip_*_lausanne/ directories (same structure as our MIP datasets).

Usage:
    python -m steps.step_mip.generate_lausanne_mips \\
        --source /path/to/lausanne_nifti \\
        --output-root /path/to/datasets
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from medprep.mip import load_volume, resize_2d, top_z_crop

from ._cli import run_variant_driver


VARIANTS = ("mip_axial", "mip_coronal", "mip_sagittal", "mip_3ch")
_PROJECTION_AXES = {"mip_axial": 2, "mip_coronal": 1, "mip_sagittal": 0}
# Output dirs use a _lausanne suffix to keep separate from our data
_LAUSANNE_VARIANTS = tuple(f"{v}_lausanne" for v in VARIANTS)


def process_volume(nifti_path: str | Path, target_size: int = 384) -> dict[str, np.ndarray]:
    """Generate 3 MIPs from a Lausanne volume with Z cropping."""
    data = top_z_crop(load_volume(nifti_path), 60)

    target = (target_size, target_size)

    mips = {}
    for name, axis in _PROJECTION_AXES.items():
        mip = np.max(data, axis=axis)
        mips[name] = resize_2d(mip, target)

    mips["mip_3ch"] = np.stack(
        [mips["mip_axial"], mips["mip_coronal"], mips["mip_sagittal"]], axis=0
    )
    return mips


def _process_lausanne_variant(fpath: Path, target_size: int) -> dict[str, np.ndarray]:
    """process_volume(), with output keys renamed to the _lausanne variant dirs."""
    mips = process_volume(fpath, target_size)
    return {f"{name}_lausanne": arr for name, arr in mips.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIPs from Lausanne volumes")
    parser.add_argument("--source", required=True, help="lausanne_nifti/ dir with 0/ and 1/")
    parser.add_argument("--output-root", required=True, help="Parent dir for MIP datasets")
    parser.add_argument("--target-size", type=int, default=384)
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output_root)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    run_variant_driver(
        source_dir,
        output_root,
        _LAUSANNE_VARIANTS,
        lambda fpath: _process_lausanne_variant(fpath, args.target_size),
        file_ext=(".nii.gz",),
        track_errors=True,
    )


if __name__ == "__main__":
    main()
