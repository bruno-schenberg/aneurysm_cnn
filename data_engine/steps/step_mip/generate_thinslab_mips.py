"""Generate thin-slab MIP datasets from original NIfTIs.

Instead of projecting the entire volume into one image, split Z axis into
N equal slabs and MIP each slab separately. This preserves depth information
that full-volume MIP discards.

Output: (N, H, W) .npy files where N = number of slabs.
Each channel is the MIP of one thin slab along the axial (Z) axis.

Usage:
    python -m steps.step_mip.generate_thinslab_mips \
        --source /path/to/nifti \
        --output-root /path/to/datasets \
        --n-slabs 5 \
        --top-crop 90
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from medprep.mip import load_volume, resize_2d, top_z_crop

from ._cli import collect_class_files, count_variant_files, ensure_variant_dirs, file_stem


def process_volume(
    nifti_path: str | Path,
    n_slabs: int,
    top_crop_pct: int = 90,
    target_size: int = 384,
) -> np.ndarray:
    """Load NIfTI, top-Z crop, split into thin slabs, MIP each.

    Returns (n_slabs, H, W) array.
    """
    data = top_z_crop(load_volume(nifti_path), top_crop_pct)

    z_cropped = data.shape[2]
    target = (target_size, target_size)

    # Split Z into n_slabs equal slabs
    slab_size = z_cropped / n_slabs
    mips = []
    for i in range(n_slabs):
        z_lo = int(round(i * slab_size))
        z_hi = int(round((i + 1) * slab_size))
        z_hi = max(z_hi, z_lo + 1)  # at least 1 slice
        slab = data[:, :, z_lo:z_hi]
        mip = np.max(slab, axis=2)  # axial MIP of this slab
        mips.append(resize_2d(mip, target))

    return np.stack(mips, axis=0)  # (n_slabs, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thin-slab axial MIP datasets"
    )
    parser.add_argument("--source", required=True, help="NIfTI dir with 0/ and 1/")
    parser.add_argument("--output-root", required=True, help="Parent dir for datasets")
    parser.add_argument("--n-slabs", type=int, default=5, help="Number of thin slabs (default: 5)")
    parser.add_argument("--top-crop", type=int, default=90, help="Top-Z crop percent (default: 90)")
    parser.add_argument("--target-size", type=int, default=384)
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output_root)

    if not source_dir.exists():
        print(f"ERROR: Source not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    variant = f"mip_thinslab{args.n_slabs}"
    ensure_variant_dirs(output_root, [variant])

    for cls in ("0", "1"):
        files = [f for f, _ in collect_class_files(source_dir, (".nii", ".nii.gz"), classes=(cls,))]

        for i, fpath in enumerate(files):
            stem = file_stem(fpath)
            out_path = output_root / variant / cls / f"{stem}.npy"
            if out_path.exists():
                continue

            result = process_volume(fpath, args.n_slabs, args.top_crop, args.target_size)
            np.save(str(out_path), result)

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  [{i+1}/{len(files)}] {stem}")

    count = count_variant_files(output_root, [variant])[variant]
    print(f"\nDone. {variant}: {count} files (shape: ({args.n_slabs}, {args.target_size}, {args.target_size}))")


if __name__ == "__main__":
    main()
