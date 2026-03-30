"""
create_sample_dataset.py

Generates a minimal synthetic NIfTI dataset for testing the training pipeline
on the cluster without the real /mnt/data dataset.

Creates:
    /mnt/data/cases-3/sample_dataset/
        0/   — 12 synthetic negative (healthy) volumes
        1/   — 12 synthetic positive (aneurysm) volumes

Each volume is 32x32x32 voxels (.nii.gz). MONAI will resize them to
128x128x128 during loading, so the size doesn't matter for the test.

Usage (run from repo root, with conda data env active):
    python hpc/singularity/create_sample_dataset.py

Then transfer to the cluster:
    scp -r /mnt/data/cases-3/sample_dataset/ <user>@drummond:~/
"""

import os
import numpy as np
import nibabel as nib

OUTPUT_DIR = "/mnt/data/cases-3/sample_dataset"
N_PER_CLASS = 12   # enough for 20% test split + 2-fold CV with stratification
SHAPE = (32, 32, 32)
RNG = np.random.default_rng(42)


def make_volume(label: int) -> np.ndarray:
    """Synthetic 32³ volume. Positives have a brighter central blob."""
    vol = RNG.integers(0, 300, size=SHAPE, dtype=np.int16)
    if label == 1:
        # Small bright sphere in the centre to make positive class distinct
        cx, cy, cz = [s // 2 for s in SHAPE]
        r = 5
        for x in range(cx - r, cx + r):
            for y in range(cy - r, cy + r):
                for z in range(cz - r, cz + r):
                    if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < r**2:
                        vol[x, y, z] = RNG.integers(800, 1200)
    return vol


def save_volume(vol: np.ndarray, path: str) -> None:
    affine = np.eye(4)  # 1mm isotropic identity affine
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, path)


def main() -> None:
    for label in [0, 1]:
        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)
        for i in range(N_PER_CLASS):
            vol = make_volume(label)
            path = os.path.join(class_dir, f"case_{label}_{i:03d}.nii.gz")
            save_volume(vol, path)
            print(f"  Saved {path}")

    total = N_PER_CLASS * 2
    print(f"\nDone — {total} volumes in '{OUTPUT_DIR}/' ({N_PER_CLASS} per class)")
    print("\nTransfer to cluster:")
    print(f"  scp -r {OUTPUT_DIR} <user>@drummond:~/")


if __name__ == "__main__":
    main()
