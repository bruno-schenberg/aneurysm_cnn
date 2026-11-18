"""Generate MIP (Maximum Intensity Projection) datasets from 3D volumes.

Creates 2D projections along 3 axes plus a 3-channel combined version.
Output datasets (under --output-root):
  mip_axial/     — max along axis 2 (Z), resized to target_size x target_size
  mip_coronal/   — max along axis 1 (Y), resized to target_size x target_size
  mip_sagittal/  — max along axis 0 (X), resized to target_size x target_size
  mip_3ch/       — 3-channel stack [axial, coronal, sagittal]

Each saved as .npy in the standard 0/ 1/ class structure.

Usage:
    python -m steps.step_mip.generate_mips --source /path/to/c90z_384 --output-root /path/to/datasets
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from medprep.mip import load_volume, resize_2d

from ._cli import run_variant_driver


VARIANTS = ("mip_axial", "mip_coronal", "mip_sagittal", "mip_3ch")
# axis index for np.max: axial=Z(2), coronal=Y(1), sagittal=X(0)
_PROJECTION_AXES = {"mip_axial": 2, "mip_coronal": 1, "mip_sagittal": 0}


def process_volume(nifti_path: str | Path, target_size: int = 384) -> dict[str, np.ndarray]:
    """Generate 3 MIPs from a single NIfTI volume.

    Returns dict with keys: mip_axial, mip_coronal, mip_sagittal, mip_3ch
    """
    data = load_volume(nifti_path)
    target = (target_size, target_size)

    mips = {}
    for name, axis in _PROJECTION_AXES.items():
        mip = np.max(data, axis=axis)
        mips[name] = resize_2d(mip, target)

    mips["mip_3ch"] = np.stack(
        [mips["mip_axial"], mips["mip_coronal"], mips["mip_sagittal"]], axis=0
    )
    return mips


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIP datasets from 3D volumes")
    parser.add_argument("--source", required=True, help="Source dataset dir with 0/ and 1/ subdirs")
    parser.add_argument("--output-root", required=True, help="Parent dir for MIP dataset dirs")
    parser.add_argument("--target-size", type=int, default=384, help="Output MIP spatial size")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output_root)
    target_size = args.target_size

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    run_variant_driver(
        source_dir,
        output_root,
        VARIANTS,
        lambda fpath: process_volume(fpath, target_size),
        skipped_label="Skipped (existing)",
    )


if __name__ == "__main__":
    main()
