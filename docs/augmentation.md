# Data Augmentation

All augmentation is applied at training time only via MONAI dictionary
transforms in `src/data.py:get_transforms()`. Validation and test sets
receive no augmentation. Every parameter is exposed as a config key in
`DEFAULT_CONFIG` and can be overridden per-experiment in the JSON file.

## Transforms

| Transform | Config keys | Default | Description |
|-----------|-------------|---------|-------------|
| **RandFlipd** (x3) | `AUG_FLIP_PROB` | 0.5 | Independent random flip along each spatial axis (0, 1, 2). Three separate transforms, one per axis. |
| **RandRotate90d** | `AUG_ROT90_PROB` | 0.5 | Discrete 90-degree rotation (k=1,2,3 random). No continuous rotation — interpolation on axis-aligned MRA volumes produces artifacts. |
| **RandScaleIntensityd** | `AUG_SCALE_INTENSITY_PROB`, `AUG_SCALE_INTENSITY_RANGE` | 0.5, 0.1 | Multiplicative intensity scaling by a random factor in [-range, +range]. Simulates acquisition intensity variation. |
| **RandZoomd** | `AUG_ZOOM_PROB`, `AUG_ZOOM_MIN`, `AUG_ZOOM_MAX` | 0.3, 0.9, 1.1 | Isotropic random zoom (90-110%) with `keep_size=True` (output padded/cropped to original shape). Replaces the dropped crop-based dataset variants — provides scale variation without generating separate datasets. |
| **RandGaussianNoised** | `AUG_NOISE_PROB`, `AUG_NOISE_STD` | 0.2, 0.05 | Additive Gaussian noise. A placeholder — Rician noise (`RandRicianNoised`, backlog) is the physically accurate MRI noise model. |

## Pipeline order

1. `LoadImaged` — load NIfTI
2. `EnsureChannelFirstd` — add channel dim: (D,H,W) -> (1,D,H,W)
3. Normalization (percentile / zscore / histogram) — deterministic, cached by `CacheDataset`
4. Augmentation transforms (training only) — randomized, re-applied every fetch
5. `EnsureTyped` — convert to PyTorch tensors

MONAI's `CacheDataset` automatically detects the first randomizable
transform and caches only the deterministic prefix (steps 1-3). Steps 4+
are re-executed on every `__getitem__` call, so augmentation varies per
epoch even with `CACHE_RATE=1.0`.

## Design decisions

- **No continuous rotation**: discrete 90-degree rotations only.
  Continuous rotation requires trilinear interpolation, which produces
  visible artifacts on axis-aligned MRA data (vessels become blurred at
  non-90-degree angles). Confirmed visually.

- **Rotation restricted to axial plane (Z-axis)**: `RandRotate90d` uses
  the default `spatial_axes=(0, 1)`, which rotates only in the axial
  plane (around Z). Cross-plane 90° rotations would swap
  axial↔coronal or axial↔sagittal, changing the relationship between
  anatomy and acquisition direction. This is the standard approach for
  brain imaging — nnU-Net (Isensee et al., 2021) and most 3D medical
  image augmentation pipelines restrict rotations to the axial plane.

- **All-axis flips**: `RandFlipd` is applied independently along each
  spatial axis (0, 1, 2) at prob=0.5. Left-right (sagittal) flipping
  is universally accepted for brain imaging due to approximate
  bilateral symmetry. Superior-inferior and anterior-posterior flips
  are more aggressive but reasonable for aneurysm detection, where
  lesions have no strong positional bias along those axes. If future
  experiments show overfitting or degraded performance, restricting to
  left-right flip only (`spatial_axis=0`) is the first thing to try.

- **Random zoom replaces crop variants**: earlier pipeline versions
  generated separate "crop" dataset variants at different scales. Random
  zoom (90-110%) achieves the same scale variation at training time,
  eliminating the need for extra datasets on disk.

- **Gaussian noise as placeholder**: the current `RandGaussianNoised` is
  a simple approximation. MRI noise follows a Rician distribution
  `I = sqrt((I + N_real)^2 + N_imag^2)`. A `RandRicianNoised` transform
  is planned (backlog) to selectively add noise to 3T volumes (~50% of
  the time) to simulate 1.5T acquisition conditions.

## Disabling augmentation

Set all `AUG_*_PROB` keys to `0.0` in the experiment config. This is
used by the overfitting test (`tests/test_overfit.py`) to verify the
training loop learns without augmentation interference.
