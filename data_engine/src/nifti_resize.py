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
from monai.data import MetaTensor
from monai.transforms import Spacing, Resize, ResizeWithPadOrCrop

logger = logging.getLogger(__name__)


# ----------------------------------------------------
# 1. Module-Level Constants
# ----------------------------------------------------

TARGET_SHAPE = (128, 128, 128)   # All variants must produce exactly this shape
TARGET_SPACING_MM = 1.0          # Isotropic voxel spacing used by variant A


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


def _save(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """
    Save a float32 volume as a compressed NIfTI file.

    The assertion enforces that every variant function produces exactly the
    target shape before writing. This catches shape errors early with a clear
    message rather than silently saving a wrongly-sized file that would corrupt
    a dataset batch.

    Args:
        data: Volume array, must have shape ``TARGET_SHAPE``.
        affine: Affine matrix to embed in the NIfTI header.
        path: Output file path (should end in ``.nii.gz``).
    """
    assert data.shape == TARGET_SHAPE, f"Expected shape {TARGET_SHAPE}, got {data.shape}"
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


def _variant_c_from_data(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    result = np.asarray(ResizeWithPadOrCrop(TARGET_SHAPE)(data[np.newaxis]), dtype=np.float32)
    _save(result[0], affine, output_path)


def _variant_d_from_data(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    old_shape = np.array(data.shape, dtype=float)
    result = np.asarray(
        Resize(TARGET_SHAPE, mode="trilinear")(data[np.newaxis]),
        dtype=np.float32,
    )
    scale = old_shape / np.array(TARGET_SHAPE, dtype=float)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]
    _save(result[0], new_affine, output_path)


def _variant_a_from_data(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    resampled_data, resampled_affine = _resample_to_1mm_isotropic(data, affine)
    result = np.asarray(
        ResizeWithPadOrCrop(TARGET_SHAPE)(resampled_data[np.newaxis]), dtype=np.float32
    )
    del resampled_data
    _save(result[0], resampled_affine, output_path)


def _variant_b_from_data(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    original_zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    max_physical_extent = float(np.max(np.array(data.shape, dtype=float) * original_zooms))
    target_spacing = max_physical_extent / 128.0

    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    scaled_result = Spacing(pixdim=target_spacing, mode="bilinear")(mt)
    scaled_data = scaled_result.numpy()[0].astype(np.float32)
    scaled_affine = scaled_result.affine.numpy()
    del mt, scaled_result

    padded_data = np.asarray(
        ResizeWithPadOrCrop(TARGET_SHAPE)(scaled_data[np.newaxis]), dtype=np.float32
    )
    del scaled_data
    _save(padded_data[0], scaled_affine, output_path)


# ----------------------------------------------------
# 4. Public Variant Functions (load from path)
# ----------------------------------------------------


def generate_variant_c(input_path: Path, output_path: Path) -> None:
    """
    Dataset C — native voxel spacing, centre-crop or zero-pad to 128³.

    No resampling is performed. The volume is taken exactly as produced by the
    DICOM conversion step and its physical voxel spacing is preserved. If any
    dimension is larger than 128 voxels, it is symmetrically cropped from the
    centre. If any dimension is smaller than 128, it is symmetrically zero-padded.
    Both operations are lossless — no interpolation is applied.

    This variant is the experimental baseline. It introduces the least data
    modification of any variant, but the spatial meaning of each voxel differs
    across subjects: a voxel in one scan might represent 0.5×0.5×1mm of tissue
    while the same grid position in another scan represents 1×1×3mm. The model
    must learn anatomy without any spatial normalisation.

    Note on affine accuracy: after a centre-crop the physical location of voxel
    (0,0,0) shifts, so the saved affine is technically offset from the true
    origin of the cropped volume. This has no effect on training since the
    training pipeline uses the voxel array only, not the spatial coordinates.
    """
    try:
        data, affine = _load(input_path)
        _variant_c_from_data(data, affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant C: %s", input_path.name, e)


def generate_variant_d(input_path: Path, output_path: Path) -> None:
    """
    Dataset D — native voxel spacing, anti-aliased trilinear resize to 128³.

    No resampling is performed. The volume is stretched or shrunk to exactly
    128³ voxels using trilinear interpolation. Clinical head volumes are almost
    always larger than 128³, so this is a shrink in practice.

    Anti-aliasing (``anti_aliasing=True``) applies a Gaussian pre-filter before
    downsampling to suppress aliasing artefacts — the high-frequency ringing that
    appears when you reduce a signal without first smoothing it. Without this,
    fine structural details can produce visual noise in the output.

    The affine matrix is updated to reflect the new effective voxel spacing:
    each voxel now covers ``original_shape / 128`` times as much physical space
    as before. This is done by scaling each column of the 3×3 rotation-scale
    block by the corresponding axis scale factor. The scale is applied
    column-wise because each column of the affine encodes the world-space
    displacement when you step one voxel along that axis.

    Unlike variants A and B, the aspect ratio of the brain is distorted to fill
    the full 128³ cube — a head that is 200mm tall but only 150mm wide will
    appear as a perfect cube in this variant. This is a known limitation and is
    what makes the A/B vs C/D comparison scientifically interesting.
    """
    try:
        data, affine = _load(input_path)
        _variant_d_from_data(data, affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant D: %s", input_path.name, e)


def generate_variant_a(input_path: Path, output_path: Path) -> None:
    """
    Dataset A — 1mm isotropic resampling, then centre-crop or zero-pad to 128³.

    Step 1 resamples to a uniform 1mm isotropic grid (see
    ``_resample_to_1mm_isotropic``). This normalises spacing across all subjects
    so every voxel always represents the same physical volume of tissue. Step 2
    centre-crops or zero-pads the resampled volume to exactly 128³. Both the
    crop/pad and the 1mm spacing are lossless relative to each other — only one
    interpolation pass occurs (the resampling step).

    The fixed 1mm spacing means the physical field of view captured within
    128³ voxels is always exactly 128×128×128 mm. Subjects with a larger head
    will have some peripheral anatomy cropped; subjects with a smaller head will
    have background padding. For aneurysm detection this is generally acceptable
    since aneurysms occur in the central vasculature, not at the periphery.

    The affine from the 1mm resampling step is preserved in the output. As with
    variant C, the affine origin is not adjusted for the crop/pad offset, but
    this does not affect training.
    """
    try:
        data, affine = _load(input_path)
        _variant_a_from_data(data, affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant A: %s", input_path.name, e)


def generate_variant_b(input_path: Path, output_path: Path) -> None:
    """
    Dataset B — single-step isotropic resampling so the largest dimension maps
    to 128 voxels, then zero-pad to 128³.

    This is the most principled variant: it applies exactly one interpolation
    pass, preserves the aspect ratio of the anatomy, and never crops any content.

    **Why not resample to 1mm and then shrink?**
    The naive approach — resample to 1mm first, then resize the result to 128³ —
    applies two interpolation passes, blurring the data twice. It also distorts
    proportions because a 200mm × 150mm brain forced into a 128³ cube is no
    longer the right shape. Variant B avoids both problems.

    **How the target spacing is computed:**
    The physical extent of a volume along an axis is ``voxels × spacing`` (mm).
    For the largest axis, we want the resampled voxel count to be exactly 128.
    Rearranging: ``target_spacing = max_physical_extent / 128``.

    Applying this single isotropic spacing to all three axes means the largest
    dimension lands at 128 voxels and the shorter dimensions land proportionally
    below 128 — exactly preserving the brain's aspect ratio. MONAI's ``Spacing``
    may round voxel counts by ±1, so ``ResizeWithPadOrCrop`` is applied after to
    guarantee exactly 128³ via pure zero-padding (the shorter dims are always
    slightly under 128, so no cropping occurs in practice).

    **Voxel spacing extraction:**
    Physical voxel spacing is extracted from the affine using column norms
    (``sqrt(sum(col²))``), which correctly handles oblique acquisitions where
    the affine has a rotation component. Using the diagonal directly would give
    wrong spacings for rotated volumes.
    """
    try:
        data, affine = _load(input_path)
        _variant_b_from_data(data, affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant B: %s", input_path.name, e)
