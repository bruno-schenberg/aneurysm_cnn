"""
nifti_resize.py

Generates the four preprocessing variants (A, B, C, D) from the clean NIfTI
volumes produced by the DICOM conversion step. Each variant represents a
different strategy for normalising spatial resolution and fitting every volume
into the fixed 128×128×128 voxel grid required by the CNN.

## Why four variants?

The CNN requires all input volumes to be the same size (128³). There are two
independent decisions to make when getting there:

  1. Whether to normalise voxel spacing first (isotropic resampling: yes/no)
  2. How to fit the volume into 128³ (crop/pad vs shrink)

These two binary choices produce a 2×2 design:

  ┌─────────────────┬──────────────────────────┬──────────────────────────────┐
  │                 │  No resampling           │  Isotropic resampling        │
  ├─────────────────┼──────────────────────────┼──────────────────────────────┤
  │  Crop / pad     │  C — native, lossless    │  A — 1mm iso + crop/pad      │
  │  Shrink         │  D — native, resize      │  B — iso resample + pad      │
  └─────────────────┴──────────────────────────┴──────────────────────────────┘

This design lets the training experiments answer two concrete questions:
  - Does normalising voxel spacing help the model generalise across scanners?
  - Does filling the full 128³ cube produce better results than preserving the
    original proportions with zero-padding?

## Why not shrink after isotropic resampling (the naive approach)?

A natural but incorrect approach would be to resample to 1mm isotropic first,
then shrink the result to 128³ with a second resize. This applies **two rounds
of interpolation** to the same data, blurring it twice. It also distorts the
aspect ratio of the brain — a head that is 200mm tall and 150mm wide gets
squashed into a perfect cube.

Variant B avoids this by computing the correct target spacing directly:
``target_spacing = max_physical_extent / 128``. A single Spacing call then
resamples the volume so its largest dimension lands at exactly 128 voxels.
The shorter dimensions are zero-padded, not squashed. One interpolation pass,
no geometry distortion.

## Interpolation count per variant

  A — 1 (bilinear resampling)      + lossless crop/pad
  B — 1 (bilinear resampling)      + lossless pad
  C — 0 (no resampling)            + lossless crop/pad
  D — 1 (trilinear resize)

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

TARGET_SHAPE = (128, 128, 128)   # Default output grid — kept for backwards compatibility
TARGET_SPACING_MM = 1.0          # Isotropic voxel spacing for variant A at 128³

# Single source of truth for all supported output grids.
# To add a new resolution: add an entry here AND a matching entry in
# OUTPUT_PATHS in dataset_gen.py. No other code changes are required.
VALID_TARGET_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "128x128x128": (128, 128, 128),
    "256x256x128": (256, 256, 128),
}

# Fixed isotropic voxel spacing used by variant A for each supported grid.
# Variant A always uses a scanner-independent fixed spacing so that one voxel
# always represents the same physical volume regardless of the source scanner.
#
# Derivation:
#   128x128x128 → 1.0 mm  (128 mm field of view at 128 voxels)
#   256x256x128 → 0.9375 mm  (240 mm typical head XY extent / 256 voxels)
#
# Variant B does NOT use this table — it computes spacing dynamically as
# max_physical_extent / target_shape[0], preserving per-scan proportions.
TARGET_SPACING_MM_BY_SHAPE: Dict[Tuple[int, int, int], float] = {
    (128, 128, 128): 1.0,
    (256, 256, 128): 0.9375,
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
            (128, 128, 128) for backwards compatibility. Pass the same value
            that was used in the calling variant function.
    """
    assert data.shape == target_shape, f"Expected shape {target_shape}, got {data.shape}"
    nib.save(nib.Nifti1Image(data, affine), path)


def _resample_to_1mm_isotropic(
    data: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a volume to 1 mm isotropic voxel spacing using MONAI's Spacing transform.

    Voxel spacing varies across scanners and acquisition protocols. A scanner
    using 0.5×0.5×3mm voxels and one using 1×1×1mm voxels represent the same
    anatomy at very different scales. Resampling to a fixed 1mm isotropic grid
    makes every scan comparable: one voxel always represents a 1mm³ cube of tissue,
    regardless of which scanner acquired it.

    MONAI's ``Spacing`` transform reads the current spacing from the affine matrix,
    computes the required scaling, applies bilinear interpolation, and returns a
    new MetaTensor with an updated affine that reflects the 1mm spacing.

    The ``pixdim=1.0`` scalar applies the same 1mm target to all three axes,
    producing a truly isotropic output. The resulting volume will have a different
    shape than the input — a scan with 0.5mm in-plane resolution and 512×512
    pixels will become a 512×512 volume at 1mm, i.e. approximately 256×256 at 1mm.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.

    Returns:
        Tuple of (resampled data, updated affine) at 1mm isotropic spacing.
    """
    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),  # MONAI expects channel-first: (1, H, W, D)
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    result = Spacing(pixdim=TARGET_SPACING_MM, mode="bilinear")(mt)
    resampled = result.numpy()[0].astype(np.float32)  # strip channel dim
    new_affine = result.affine.numpy()
    del mt, result
    return resampled, new_affine


# ----------------------------------------------------
# 3. Per-Variant Transform Helpers (accept pre-loaded data)
# ----------------------------------------------------
#
# These functions contain the transform logic for each variant but do not
# load or save files themselves. They receive a pre-loaded (data, affine)
# pair so the caller can load the file once and reuse it across all variants,
# avoiding 4 separate disk reads and 4 rounds of allocator churn per file.
#
# The public generate_variant_X functions below are thin wrappers that load
# from a path, call the corresponding helper, then delegate saving.


def _variant_c_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant C: native spacing, centre-crop or zero-pad to ``target_shape``. No interpolation.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
            Pass ``(256, 256, 128)`` to produce the higher-resolution variant.
    """
    result = np.asarray(ResizeWithPadOrCrop(target_shape)(data[np.newaxis]), dtype=np.float32)
    _save(result[0], affine, output_path, target_shape=target_shape)


def _variant_d_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant D: native spacing, trilinear resize to ``target_shape``. Affine scaled to match new voxel size.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
            The affine is updated so each voxel in the output covers
            ``original_shape / target_shape`` times the original physical space.
    """
    old_shape = np.array(data.shape, dtype=float)
    result = np.asarray(
        Resize(target_shape, mode="trilinear")(data[np.newaxis]),
        dtype=np.float32,
    )
    scale = old_shape / np.array(target_shape, dtype=float)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]
    _save(result[0], new_affine, output_path, target_shape=target_shape)


def _variant_a_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant A: resample to fixed isotropic spacing, then centre-crop or zero-pad to ``target_shape``.

    The target spacing is looked up from ``TARGET_SPACING_MM_BY_SHAPE`` using
    ``target_shape`` as the key. For 128³ this is 1.0 mm; for 256×256×128 this
    is 0.9375 mm (derived as 240 mm typical head XY extent / 256 voxels).

    **Why a fixed spacing instead of dynamic (like variant B)?**
    Variant A's scientific purpose is scanner independence: one voxel always
    represents the same physical volume regardless of the source scanner. A
    dynamic spacing would vary between subjects and defeat this goal.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
    """
    spacing = TARGET_SPACING_MM_BY_SHAPE[target_shape]
    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    result_mt = Spacing(pixdim=spacing, mode="bilinear")(mt)
    resampled_data = result_mt.numpy()[0].astype(np.float32)
    resampled_affine = result_mt.affine.numpy()
    del mt, result_mt
    result = np.asarray(
        ResizeWithPadOrCrop(target_shape)(resampled_data[np.newaxis]), dtype=np.float32
    )
    del resampled_data
    _save(result[0], resampled_affine, output_path, target_shape=target_shape)


def _variant_b_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """Variant B: single-pass isotropic resample (largest dim → ``target_shape[0]`` voxels), then zero-pad.

    **How target spacing is computed:**
    ``target_spacing = max_physical_extent / target_shape[0]``
    where ``max_physical_extent = max(voxel_count × voxel_spacing)`` over all
    three axes. This maps the largest physical dimension to exactly
    ``target_shape[0]`` voxels in one resampling pass, preserving the anatomy's
    aspect ratio. Shorter dimensions land proportionally below the target and
    are zero-padded by ``ResizeWithPadOrCrop``.

    **Why not use a fixed spacing (like variant A)?**
    Variant B's purpose is to avoid double interpolation and aspect-ratio
    distortion. A fixed spacing would require a second resize step. Computing
    the spacing dynamically achieves the goal in a single pass.

    Args:
        data: Input volume array shape (H, W, D).
        affine: Affine matrix encoding the current voxel spacing.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
            For ``(256, 256, 128)`` the divisor becomes 256, naturally producing
            ~0.938 mm spacing for clinical brain scans.
    """
    original_zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    max_physical_extent = float(np.max(np.array(data.shape, dtype=float) * original_zooms))
    target_spacing = max_physical_extent / float(target_shape[0])

    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    scaled_result = Spacing(pixdim=target_spacing, mode="bilinear")(mt)
    scaled_data = scaled_result.numpy()[0].astype(np.float32)
    scaled_affine = scaled_result.affine.numpy()
    del mt, scaled_result

    padded_data = np.asarray(
        ResizeWithPadOrCrop(target_shape)(scaled_data[np.newaxis]), dtype=np.float32
    )
    del scaled_data
    _save(padded_data[0], scaled_affine, output_path, target_shape=target_shape)


# ----------------------------------------------------
# 4. Public Variant Functions (load from path)
# ----------------------------------------------------


def generate_variant_c(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """
    Dataset C — native voxel spacing, centre-crop or zero-pad to ``target_shape``.

    No resampling is performed. The volume is taken exactly as produced by the
    DICOM conversion step and its physical voxel spacing is preserved. If any
    dimension is larger than the corresponding ``target_shape`` dimension, it is
    symmetrically cropped from the centre. If any dimension is smaller, it is
    symmetrically zero-padded. Both operations are lossless — no interpolation
    is applied.

    This variant is the experimental baseline. It introduces the least data
    modification of any variant, but the spatial meaning of each voxel differs
    across subjects: a voxel in one scan might represent 0.5×0.5×1mm of tissue
    while the same grid position in another scan represents 1×1×3mm. The model
    must learn anatomy without any spatial normalisation.

    Note on affine accuracy: after a centre-crop the physical location of voxel
    (0,0,0) shifts, so the saved affine is technically offset from the true
    origin of the cropped volume. This has no effect on training since the
    training pipeline uses the voxel array only, not the spatial coordinates.

    Args:
        input_path: Source ``.nii.gz`` file.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
    """
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
    """
    Dataset D — native voxel spacing, trilinear resize to ``target_shape``.

    No resampling is performed. The volume is stretched or shrunk to exactly
    ``target_shape`` voxels using trilinear interpolation. Clinical head volumes
    are almost always larger than 128³, so this is a shrink in practice.

    The affine matrix is updated to reflect the new effective voxel spacing:
    each voxel now covers ``original_shape / target_shape`` times as much
    physical space as before. This is done by scaling each column of the 3×3
    rotation-scale block by the corresponding axis scale factor. The scale is
    applied column-wise because each column of the affine encodes the world-space
    displacement when you step one voxel along that axis.

    Unlike variants A and B, the aspect ratio of the brain is distorted to fill
    the full target cube — a head that is 200mm tall but only 150mm wide will
    appear as a perfect cube in this variant. This is a known limitation and is
    what makes the A/B vs C/D comparison scientifically interesting.

    Args:
        input_path: Source ``.nii.gz`` file.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
    """
    try:
        data, affine = _load(input_path)
        _variant_d_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant D: %s", input_path.name, e)


def generate_variant_a(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None:
    """
    Dataset A — fixed isotropic resampling, then centre-crop or zero-pad to ``target_shape``.

    Step 1 resamples to a fixed isotropic grid whose spacing is looked up from
    ``TARGET_SPACING_MM_BY_SHAPE`` (1.0 mm for 128³, 0.9375 mm for 256×256×128).
    This normalises spacing across all subjects so every voxel always represents
    the same physical volume of tissue. Step 2 centre-crops or zero-pads the
    resampled volume to exactly ``target_shape``. Only one interpolation pass
    occurs (the resampling step).

    The fixed spacing means the physical field of view captured within
    ``target_shape`` voxels is always the same regardless of the source scanner.
    Subjects with a larger head will have some peripheral anatomy cropped;
    subjects with a smaller head will have background padding. For aneurysm
    detection this is generally acceptable since aneurysms occur in the central
    vasculature, not at the periphery.

    The affine from the resampling step is preserved in the output. As with
    variant C, the affine origin is not adjusted for the crop/pad offset, but
    this does not affect training.

    Args:
        input_path: Source ``.nii.gz`` file.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
    """
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
    """
    Dataset B — single-step isotropic resampling so the largest dimension maps
    to ``target_shape[0]`` voxels, then zero-pad to ``target_shape``.

    This is the most principled variant: it applies exactly one interpolation
    pass, preserves the aspect ratio of the anatomy, and never crops any content.

    **Why not resample to a fixed spacing and then shrink?**
    A fixed spacing + second resize applies two interpolation passes, blurring
    the data twice. It also distorts proportions because a 200mm × 150mm brain
    forced into a cube is no longer the right shape. Variant B avoids both problems.

    **How the target spacing is computed:**
    The physical extent of a volume along an axis is ``voxels × spacing`` (mm).
    For the largest axis, we want the resampled voxel count to be exactly
    ``target_shape[0]``. Rearranging: ``target_spacing = max_physical_extent / target_shape[0]``.

    Applying this single isotropic spacing to all three axes means the largest
    dimension lands at ``target_shape[0]`` voxels and the shorter dimensions land
    proportionally below — exactly preserving the brain's aspect ratio. MONAI's
    ``Spacing`` may round voxel counts by ±1, so ``ResizeWithPadOrCrop`` is
    applied after to guarantee exactly ``target_shape`` via pure zero-padding.

    **Voxel spacing extraction:**
    Physical voxel spacing is extracted from the affine using column norms
    (``sqrt(sum(col²))``), which correctly handles oblique acquisitions where
    the affine has a rotation component. Using the diagonal directly would give
    wrong spacings for rotated volumes.

    Args:
        input_path: Source ``.nii.gz`` file.
        output_path: Destination ``.nii.gz`` file path.
        target_shape: Output voxel grid. Defaults to ``TARGET_SHAPE`` (128, 128, 128).
            For ``(256, 256, 128)`` the divisor becomes 256, naturally producing
            ~0.938 mm spacing for clinical brain scans.
    """
    try:
        data, affine = _load(input_path)
        _variant_b_from_data(data, affine, output_path, target_shape=target_shape)
    except Exception as e:
        logger.error("Error processing %s for variant B: %s", input_path.name, e)
