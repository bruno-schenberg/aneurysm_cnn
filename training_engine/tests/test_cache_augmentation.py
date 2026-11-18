"""
Validation check: confirm CacheDataset re-applies random
transforms per fetch instead of caching augmented output.

Run: python training_engine/tests/test_cache_augmentation.py
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from monai.data import CacheDataset

from src.data import get_transforms


def demo() -> None:
    config = {
        "normalization": "percentile",
        "AUG_FLIP_PROB": 1.0,
        "AUG_ROT90_PROB": 1.0,
        "AUG_SCALE_INTENSITY_PROB": 1.0,
        "AUG_SCALE_INTENSITY_RANGE": 0.1,
        "AUG_ZOOM_PROB": 0.0,  # keep_size zoom is order-sensitive; skip for this check
        "AUG_ZOOM_MIN": 0.9,
        "AUG_ZOOM_MAX": 1.1,
        "AUG_NOISE_PROB": 1.0,
        "AUG_NOISE_STD": 0.05,
    }

    with tempfile.TemporaryDirectory() as tmp:
        vol_path = Path(tmp) / "vol.nii.gz"
        rng = np.random.default_rng(0)
        vol = rng.random((16, 16, 16), dtype=np.float32) * 100
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(vol_path))

        transform = get_transforms(config, augment=True)
        ds = CacheDataset(
            data=[{"image": str(vol_path)}],
            transform=transform,
            cache_rate=1.0,
            num_workers=0,
        )

        out_a = ds[0]["image"].numpy()
        out_b = ds[0]["image"].numpy()

        assert not np.array_equal(out_a, out_b), (
            "CacheDataset returned identical output on repeated fetch — "
            "random transforms are being cached instead of re-applied. "
            "Split get_transforms into deterministic/random pipelines."
        )
        print("OK: CacheDataset re-applies random augmentation per fetch "
              "(deterministic prefix is cached, as expected).")


if __name__ == "__main__":
    demo()
