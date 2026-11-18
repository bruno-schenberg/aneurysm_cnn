"""
Synthetic signal sanity check.

Adds a bright ~5-voxel sphere near the center of the brain to every NIfTI
volume in --input-dir, and writes the result to --output-dir. Never touches
the source files.

Usage:
    cd training_engine
    source .venv/bin/activate
    python scripts/add_synthetic_sphere.py \
        --input-dir /mnt/data/datasets/iso_128/1 \
        --output-dir /mnt/data/datasets/iso_128_marked/1

To actually train on the result, point a dataset dir's "1" class at the
marked volumes and its "0" class at the untouched originals, e.g.:
    ln -sfn /mnt/data/datasets/iso_128/0 /mnt/data/datasets/iso_128_marked/0
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data import SPHERE_RADIUS, sample_synthetic_sphere_mask, synthetic_sphere_seed


def add_sphere(data: np.ndarray, seed: int, path: str) -> np.ndarray:
    """Return a copy of data with a bright sphere near the center, inside the brain.

    Same core math and seeding as src.data.AddSyntheticSphered (the
    equivalent MONAI transform used during training) — this script exists to
    pre-bake the same sanity-check volumes to disk instead of transforming
    on the fly.
    """
    rng = np.random.default_rng(synthetic_sphere_seed(seed, path))
    sphere_mask = sample_synthetic_sphere_mask(data, rng, SPHERE_RADIUS)
    out = data.copy()
    out[sphere_mask] = data.max()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in args.input_dir.iterdir()
        if p.name.endswith((".nii", ".nii.gz"))
    )
    print(f"Found {len(files)} volumes in {args.input_dir}")

    for path in files:
        out_path = args.output_dir / path.name
        if out_path.exists():
            continue
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        marked = add_sphere(data, args.seed, str(path))
        nib.save(nib.Nifti1Image(marked, img.affine, img.header), str(out_path))

    print(f"Done. Marked volumes written to {args.output_dir}")


if __name__ == "__main__":
    main()
