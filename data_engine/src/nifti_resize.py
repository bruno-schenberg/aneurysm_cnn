"""
nifti_resize.py

Generates the six preprocessing variants (A–F) from the clean NIfTI volumes
produced by the DICOM conversion step.

## 3×2 factorial design

Three resampling strategies × two centre-crop strategies:

  ┌──────────────────────┬──────────────────────────┬──────────────────────────────┐
  │                      │  80% centre crop + fit   │  Full volume, fit only       │
  ├──────────────────────┼──────────────────────────┼──────────────────────────────┤
  │  No resampling       │  C — crop then resize    │  D — resize only             │
  │  Dynamic resampling  │  A — crop then iso pad   │  B — iso resample + pad      │
  │  Fixed resampling    │  E — crop then fixed pad │  F — fixed resample + pad    │
  └──────────────────────┴──────────────────────────┴──────────────────────────────┘

### Scientific questions

  Rows (crop vs no-crop):   A vs B,  C vs D,  E vs F
    → Does focusing on the central 80% of the anatomy improve classification?

  Columns (resampling):      C/D  vs  A/B  vs  E/F
    → Does isotropic spacing normalisation help generalisation across scanners?
    → Does a fixed (scanner-independent) spacing outperform a per-scan dynamic spacing?

## Variant details

**A** — 80% centre crop of native volume → dynamic isotropic resample so the
  cropped volume fits exactly in target_shape (no further cropping) → zero-pad.
  One interpolation pass. Dynamic spacing is per-exam: spacing = max_i(cropped_extent_i / target_i).

**B** — Dynamic isotropic resample of full volume so the largest dimension maps
  to the target without cropping → zero-pad.
  One interpolation pass. spacing = max_i(full_extent_i / target_i).

**C** — 80% centre crop of native volume → trilinear resize to target_shape.
  One interpolation pass (the resize; crop is lossless). Affine updated for
  the new effective voxel size.

**D** — Trilinear resize of full native volume to target_shape.
  One interpolation pass. Affine updated for the new effective voxel size.

**E** — 80% centre crop → fixed isotropic resample (E_SPACING_MM_BY_SHAPE) → zero-pad.
  Spacing chosen so the 80%-cropped version of the largest exam (240×0.8=192 mm)
  fits exactly in the target, guaranteeing no crop. One interpolation pass.

**F** — Fixed isotropic resample of full volume (F_SPACING_MM_BY_SHAPE) → zero-pad.
  Spacing chosen so the full largest exam (240 mm) fits exactly in the target,
  guaranteeing no crop. Scanner-independent: one voxel always represents the same
  physical volume regardless of the source scanner. One interpolation pass.

## Target shapes

  192×192×128 — 1.5:1 XY:Z ratio (small variant, matches 1.467 dataset mean)
  256×256×176 — 1.455:1 XY:Z ratio (large variant, closest match at high res)
  128×128×128 — legacy cubic target (backward compatibility only, not tested)

Called by dataset_gen.py, one call per file per variant.
"""

import logging
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from typing import Dict, Tuple
from monai.data import MetaTensor
from monai.transforms import Spacing, Resize, ResizeWithPadOrCrop

logger = logging.getLogger(__name__)


# ----------------------------------------------------
# 1. Module-Level Constants
# ----------------------------------------------------

TARGET_SHAPE = (128, 128, 128)   # Legacy default — kept for backwards compatibility

# Fraction of each native axis to retain in the centre-crop step (variants A, C, E).
CROP_FRACTION: float = 0.8

# All supported output grids.
# To add a resolution: add an entry here AND a matching entry in OUTPUT_PATHS
# in dataset_gen.py. No other code changes are required.
VALID_TARGET_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "128x128x128": (128, 128, 128),   # legacy — backward compat only
    "192x192x128": (192, 192, 128),   # 1.5:1 XY:Z ratio (small)
    "256x256x176": (256, 256, 176),   # 1.455:1 XY:Z ratio (large)
}

# Fixed isotropic spacing for variant E (80%-cropped volumes, no-crop guarantee).
# Derivation: largest 80%-cropped XY FOV = 240 mm × 0.8 = 192 mm.
# spacing = 192 mm / target_XY ensures every cropped exam fits within target_XY.
E_SPACING_MM_BY_SHAPE: Dict[Tuple[int, int, int], float] = {
    (128, 128, 128): 192.0 / 128,   # 1.500 mm
    (192, 192, 128): 192.0 / 192,   # 1.000 mm
    (256, 256, 176): 192.0 / 256,   # 0.750 mm
}

# Fixed isotropic spacing for variant F (full volumes, no-crop guarantee).
# Derivation: largest full XY FOV in dataset = 240 mm.
# spacing = 240 mm / target_XY ensures every full exam fits within target_XY.
F_SPACING_MM_BY_SHAPE: Dict[Tuple[int, int, int], float] = {
    (128, 128, 128): 240.0 / 128,   # 1.875 mm
    (192, 192, 128): 240.0 / 192,   # 1.250 mm
    (256, 256, 176): 240.0 / 256,   # 0.9375 mm
}


# ----------------------------------------------------
# 2. Private Helpers
# ----------------------------------------------------


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file and return the voxel data and affine matrix.

    ``mmap=False`` forces NiBabel to read the full file into RAM immediately
    rather than memory-mapping it. Memory-mapped files keep a file handle open
    and can cause problems when this function is called from parallel worker
    processes, where the same file might be accessed by multiple processes
    simultaneously. Loading fully into RAM avoids that.

    ``get_fdata(dtype=np.float32)`` casts to float32 on load rather than the
    default float64, halving the memory footprint of large volumes.

    Args:
        path: Path to the ``.nii.gz`` file.

    Returns:
        Tuple of (data array shape (H, W, D), affine matrix shape (4, 4)).
    """
    img = nib.load(path, mmap=False)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def _save(
    data: np.ndarray,
    affine: np.ndarray,
    path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """
    Save a float32 volume as a compressed NIfTI file.

    The assertion enforces that every variant function produces exactly the
    requested ``target_shape`` before writing. This catches shape errors early
    with a clear message rather than silently saving a wrongly-sized file that
    would corrupt a dataset batch.

    Args:
        data: Volume array, must have shape equal to ``target_shape``.
        affine: Affine matrix to embed in the NIfTI header.
        path: Output file path (should end in ``.nii.gz``).
        target_shape: Expected output shape. Defaults to ``TARGET_SHAPE``
            (128, 128, 128) for backwards compatibility.
    """
    assert data.shape == target_shape, f"Expected shape {target_shape}, got {data.shape}"
    nib.save(nib.Nifti1Image(data, affine), path)


def _center_crop(
    data: np.ndarray, crop_fraction: float = CROP_FRACTION
) -> np.ndarray:
    """
    Return the centre ``crop_fraction`` of each axis, losslessly.

    Uses ``ResizeWithPadOrCrop`` which performs pure array slicing (no
    interpolation) when the target is smaller than the input.

    Args:
        data: Input volume array shape (H, W, D).
        crop_fraction: Fraction of each axis to retain (default: ``CROP_FRACTION`` = 0.8).

    Returns:
        Cropped array with shape ``(int(H*f), int(W*f), int(D*f))``.
    """
    crop_shape = tuple(int(s * crop_fraction) for s in data.shape)
    result = np.asarray(
        ResizeWithPadOrCrop(crop_shape)(data[np.newaxis]), dtype=np.float32
    )
    return result[0]


def _dynamic_spacing(
    data: np.ndarray, affine: np.ndarray, target_shape: Tuple[int, int, int]
) -> float:
    """
    Compute the isotropic spacing that maps the largest physical axis to the
    corresponding target dimension without cropping.

    ``spacing = max_i(physical_extent_i / target_i)``

    This guarantees the resampled volume fits within ``target_shape`` on every
    axis. Smaller axes land proportionally below the target and are zero-padded
    by ``ResizeWithPadOrCrop``.

    Args:
        data: Volume array whose physical extents are to be matched.
        affine: Affine matrix encoding per-axis voxel spacing.
        target_shape: Output voxel grid.

    Returns:
        Isotropic target spacing in mm.
    """
    zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    extents = np.array(data.shape, dtype=float) * zooms
    ratios = extents / np.array(target_shape, dtype=float)
    return float(np.max(ratios))


def _resample_isotropic(
    data: np.ndarray, affine: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a volume to the given isotropic spacing using bilinear interpolation.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        spacing: Target isotropic voxel spacing in mm.

    Returns:
        Tuple of (resampled data, updated affine).
    """
    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    result = Spacing(pixdim=spacing, mode="bilinear")(mt)
    resampled = result.numpy()[0].astype(np.float32)
    new_affine = result.affine.numpy()
    del mt, result
    return resampled, new_affine


# ----------------------------------------------------
# 3. Per-Variant Transform Helpers (accept pre-loaded data)
# ----------------------------------------------------
#
# These functions contain the transform logic for each variant but do not
# load or save files themselves. They receive a pre-loaded (data, affine)
# pair so the caller can load the file once and reuse it across all variants.
#
# The public generate_variant_X functions below are thin wrappers that load
# from a path, call the corresponding helper, then delegate saving.


def _variant_a_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant A: 80% centre crop → dynamic isotropic resample → zero-pad to target_shape.

    The dynamic spacing is computed on the cropped volume so the cropped anatomy
    fits exactly within target_shape (no second crop). The result is identical
    in structure to variant B but operates on an 80%-cropped sub-volume.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE``.
    """
    cropped = _center_crop(data)
    spacing = _dynamic_spacing(cropped, affine, target_shape)
    resampled, new_affine = _resample_isotropic(cropped, affine, spacing)
    del cropped
    result = np.asarray(
        ResizeWithPadOrCrop(target_shape)(resampled[np.newaxis]), dtype=np.float32
    )
    del resampled
    _save(result[0], new_affine, output_path, target_shape=target_shape)


def _variant_b_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant B: dynamic isotropic resample of full volume → zero-pad to target_shape.

    spacing = max_i(physical_extent_i / target_i) — guarantees no cropping on any axis.
    The resampled volume is then zero-padded to exactly target_shape.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE``.
    """
    spacing = _dynamic_spacing(data, affine, target_shape)
    resampled, new_affine = _resample_isotropic(data, affine, spacing)
    result = np.asarray(
        ResizeWithPadOrCrop(target_shape)(resampled[np.newaxis]), dtype=np.float32
    )
    del resampled
    _save(result[0], new_affine, output_path, target_shape=target_shape)


def _variant_c_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant C: 80% centre crop (lossless) → trilinear resize to target_shape.

    The centre crop retains the central 80% of each native axis, discarding
    peripheral skull and air. The result is then shrunk to target_shape via
    trilinear interpolation (one interpolation pass). The affine is updated to
    reflect the new effective voxel size after the resize.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE``.
    """
    crop_shape = tuple(int(s * CROP_FRACTION) for s in data.shape)
    cropped = _center_crop(data)
    result = np.asarray(
        Resize(target_shape, mode="trilinear")(cropped[np.newaxis]), dtype=np.float32
    )
    del cropped
    # Scale affine columns: each output voxel now covers crop_shape/target_shape
    # times the original physical spacing along that axis.
    scale = np.array(crop_shape, dtype=float) / np.array(target_shape, dtype=float)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale
    _save(result[0], new_affine, output_path, target_shape=target_shape)


def _variant_d_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant D: trilinear resize of full native volume to target_shape. Affine scaled to match.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE``.
    """
    old_shape = np.array(data.shape, dtype=float)
    result = np.asarray(
        Resize(target_shape, mode="trilinear")(data[np.newaxis]),
        dtype=np.float32,
    )
    scale = old_shape / np.array(target_shape, dtype=float)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale
    _save(result[0], new_affine, output_path, target_shape=target_shape)


def _variant_e_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant E: 80% centre crop → fixed isotropic resample (E_SPACING_MM_BY_SHAPE) → zero-pad.

    The fixed spacing is derived so the 80%-cropped version of the largest exam
    (240 mm × 0.8 = 192 mm) fits exactly in target_XY — guaranteeing no further
    cropping for any exam. Equivalent to F but applied to the cropped sub-volume.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE``.
    """
    cropped = _center_crop(data)
    spacing = E_SPACING_MM_BY_SHAPE[target_shape]
    resampled, new_affine = _resample_isotropic(cropped, affine, spacing)
    del cropped
    result = np.asarray(
        ResizeWithPadOrCrop(target_shape)(resampled[np.newaxis]), dtype=np.float32
    )
    del resampled
    _save(result[0], new_affine, output_path, target_shape=target_shape)


def _variant_f_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant F: fixed isotropic resample of full volume (F_SPACING_MM_BY_SHAPE) → zero-pad.

    The fixed spacing is derived so the full largest exam (240 mm XY) fits within
    target_XY — guaranteeing no cropping. Scanner-independent: every voxel
    always represents the same physical volume of tissue regardless of scanner.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE``.
    """
    spacing = F_SPACING_MM_BY_SHAPE[target_shape]
    resampled, new_affine = _resample_isotropic(data, affine, spacing)
    result = np.asarray(
        ResizeWithPadOrCrop(target_shape)(resampled[np.newaxis]), dtype=np.float32
    )
    del resampled
    _save(result[0], new_affine, output_path, target_shape=target_shape)


# ----------------------------------------------------
# 4. Public Variant Functions (load from path)
# ----------------------------------------------------


def generate_variant_a(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Dataset A — 80% centre crop, dynamic isotropic resample, zero-pad to target_shape."""
    try:
        data, affine = _load(input_path)
        _variant_a_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant A: %s", input_path.name, e)


def generate_variant_b(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Dataset B — dynamic isotropic resample of full volume, zero-pad to target_shape."""
    try:
        data, affine = _load(input_path)
        _variant_b_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant B: %s", input_path.name, e)


def generate_variant_c(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Dataset C — 80% centre crop (lossless), trilinear resize to target_shape."""
    try:
        data, affine = _load(input_path)
        _variant_c_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant C: %s", input_path.name, e)


def generate_variant_d(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Dataset D — trilinear resize of full native volume to target_shape."""
    try:
        data, affine = _load(input_path)
        _variant_d_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant D: %s", input_path.name, e)


def generate_variant_e(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Dataset E — 80% centre crop, fixed isotropic resample, zero-pad to target_shape."""
    try:
        data, affine = _load(input_path)
        _variant_e_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant E: %s", input_path.name, e)


def generate_variant_f(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Dataset F — fixed isotropic resample of full volume, zero-pad to target_shape."""
    try:
        data, affine = _load(input_path)
        _variant_f_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant F: %s", input_path.name, e)
