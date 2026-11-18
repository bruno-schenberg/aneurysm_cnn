"""Generate multi-angle MIP datasets from original NIfTIs.

Rotates the volume around the Y axis (anterior-posterior) at multiple angles
and generates an axial MIP at each angle. This captures 3D vascular structure
from many viewpoints, like rotational angiography.

Output: (N_angles, H, W) .npy files.

Usage:
    python -m steps.step_mip.generate_multiangle_mips \
        --source /path/to/nifti \
        --output-root /path/to/datasets \
        --n-angles 6 \
        --top-crop 90
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate

from medprep.mip import load_volume, resize_2d, top_z_crop

from ._cli import collect_class_files, count_variant_files, ensure_variant_dirs, file_stem


def process_volume(
    nifti_path: str | Path,
    n_angles: int,
    top_crop_pct: int = 90,
    target_size: int = 384,
) -> np.ndarray:
    """Load NIfTI, top-Z crop, rotate at N angles, MIP each.

    Rotation is around the Y axis (axis 1) in the XZ plane.
    Returns (n_angles, H, W) array.
    """
    data = top_z_crop(load_volume(nifti_path), top_crop_pct)

    target = (target_size, target_size)
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    mips = []
    for angle in angles:
        if angle == 0:
            rotated = data
        else:
            # Rotate in the XZ plane (axes 0 and 2) around Y axis
            rotated = rotate(data, angle, axes=(0, 2), reshape=False, order=1, mode="constant", cval=0)
        # Axial MIP (max along Z = axis 2)
        mip = np.max(rotated, axis=2)
        mips.append(resize_2d(mip, target))

    return np.stack(mips, axis=0)  # (n_angles, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-angle MIP datasets"
    )
    parser.add_argument("--source", required=True, help="NIfTI dir with 0/ and 1/")
    parser.add_argument("--output-root", required=True, help="Parent dir for datasets")
    parser.add_argument("--n-angles", type=int, default=6, help="Number of rotation angles (default: 6)")
    parser.add_argument("--top-crop", type=int, default=90, help="Top-Z crop percent (default: 90)")
    parser.add_argument("--target-size", type=int, default=384)
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output_root)

    if not source_dir.exists():
        print(f"ERROR: Source not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    variant = f"mip_multiangle{args.n_angles}"
    ensure_variant_dirs(output_root, [variant])

    for cls in ("0", "1"):
        files = [f for f, _ in collect_class_files(source_dir, (".nii", ".nii.gz"), classes=(cls,))]

        for i, fpath in enumerate(files):
            stem = file_stem(fpath)
            out_path = output_root / variant / cls / f"{stem}.npy"
            if out_path.exists():
                continue

            result = process_volume(fpath, args.n_angles, args.top_crop, args.target_size)
            np.save(str(out_path), result)

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  [{i+1}/{len(files)}] {stem}")

    count = count_variant_files(output_root, [variant])[variant]
    print(f"\nDone. {variant}: {count} files (shape: ({args.n_angles}, {args.target_size}, {args.target_size}))")


if __name__ == "__main__":
    main()
