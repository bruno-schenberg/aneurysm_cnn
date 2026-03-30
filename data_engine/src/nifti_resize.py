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

TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING_MM = 1.0


# ----------------------------------------------------
# 2. Private Helpers
# ----------------------------------------------------


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return float32 data and affine without memory mapping."""
    img = nib.load(path, mmap=False)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def _save(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a float32 array as NIfTI after asserting it matches TARGET_SHAPE."""
    assert data.shape == TARGET_SHAPE, f"Expected shape {TARGET_SHAPE}, got {data.shape}"
    nib.save(nib.Nifti1Image(data, affine), path)


def _resample_to_1mm_isotropic(
    data: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a volume to 1 mm isotropic voxel spacing via MONAI Spacing (bilinear)."""
    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),  # channel-first: (1, H, W, D)
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    result = Spacing(pixdim=TARGET_SPACING_MM, mode="bilinear")(mt)
    resampled = result.numpy()[0].astype(np.float32)
    new_affine = result.affine.numpy()
    del mt, result
    return resampled, new_affine


# ----------------------------------------------------
# 3. Variant Functions
# ----------------------------------------------------


def generate_variant_c(input_path: Path, output_path: Path) -> None:
    """Centre-crop or zero-pad to 128³ at native resolution.

    No resampling is performed; the native voxel spacing is preserved. Volumes
    larger than 128³ on any axis are cropped symmetrically from the centre.
    Volumes smaller than 128³ on any axis are symmetrically zero-padded. This
    variant retains full spatial fidelity for any volume that fits within 128³,
    at the cost of inconsistent field-of-view across subjects with different
    original acquisition fields of view.
    """
    try:
        data, affine = _load(input_path)
        result = np.asarray(ResizeWithPadOrCrop(TARGET_SHAPE)(data[np.newaxis]), dtype=np.float32)
        del data
        _save(result[0], affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant C: %s", input_path.name, e)


def generate_variant_d(input_path: Path, output_path: Path) -> None:
    """Anti-aliased trilinear resize to 128³ at native resolution.

    Stretches or shrinks the volume to exactly 128³ voxels without prior
    resampling. Trilinear interpolation with a Gaussian anti-aliasing pre-filter
    prevents aliasing artefacts when shrinking — the typical case because clinical
    head MRI volumes are almost always larger than 128³. Effective voxel spacing
    changes proportionally to the ratio of original to target shape along each
    axis; the affine is updated accordingly so header and affine remain consistent.
    """
    try:
        data, affine = _load(input_path)
        old_shape = np.array(data.shape, dtype=float)
        result = np.asarray(
            Resize(TARGET_SHAPE, mode="trilinear", anti_aliasing=True)(data[np.newaxis]),
            dtype=np.float32,
        )
        del data
        scale = old_shape / np.array(TARGET_SHAPE, dtype=float)
        new_affine = affine.copy()
        new_affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]
        _save(result[0], new_affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant D: %s", input_path.name, e)


def generate_variant_a(input_path: Path, output_path: Path) -> None:
    """Resample to 1 mm isotropic voxels, then centre-crop or zero-pad to 128³.

    Step 1 normalises spatial resolution across subjects to a fixed 1 mm isotropic
    grid via bilinear interpolation, eliminating anisotropy and scale differences.
    Step 2 crops from the centre if the 1 mm volume exceeds 128³, or symmetrically
    zero-pads if it is smaller. The result is a 128³ volume at a uniform, known
    voxel spacing of 1 mm — the most straightforward variant for models that are
    sensitive to absolute physical scale.
    """
    try:
        data, affine = _load(input_path)
        data, affine = _resample_to_1mm_isotropic(data, affine)
        result = np.asarray(ResizeWithPadOrCrop(TARGET_SHAPE)(data[np.newaxis]), dtype=np.float32)
        del data
        _save(result[0], affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant A: %s", input_path.name, e)


def generate_variant_b(input_path: Path, output_path: Path) -> None:
    """Resample to 1 mm isotropic voxels, then anti-aliased trilinear resize to 128³.

    Step 1 normalises spatial resolution to 1 mm isotropic via bilinear
    interpolation, removing anisotropy. Step 2 rescales the resampled volume to
    exactly 128³ using trilinear interpolation with anti-aliasing to suppress
    artefacts during downsampling. Unlike variant A, every subject fills the full
    128³ cube with no zero-padded background. The affine from the 1 mm resampling
    step is retained so the output reports a nominal spacing of 1 mm.
    """
    try:
        data, affine = _load(input_path)
        data, affine = _resample_to_1mm_isotropic(data, affine)
        result = np.asarray(
            Resize(TARGET_SHAPE, mode="trilinear", anti_aliasing=True)(data[np.newaxis]),
            dtype=np.float32,
        )
        del data
        # The 1 mm affine is preserved as the nominal spacing for this variant;
        # the resize is treated as a fixed-grid resampling artefact, not a
        # physical scale change.
        _save(result[0], affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant B: %s", input_path.name, e)


def generate_variant_e(input_path: Path, output_path: Path) -> None:
    """Resample isotropically so the largest dimension maps to 128 voxels, then pad to 128³.

    Combines the former two-step pipeline (resample to min_zoom → scale max-dim
    to 128) into a single MONAI Spacing call, eliminating the large intermediate
    array that the first step previously produced.  The target spacing is derived
    as max_physical_extent / 128, which is algebraically equivalent to the
    original approach while preserving aspect ratio and avoiding any cropping.
    """
    try:
        data, affine = _load(input_path)
        original_zooms = np.abs(np.diagonal(affine)[:3])
        max_physical_extent = float(np.max(np.array(data.shape, dtype=float) * original_zooms))
        target_spacing = max_physical_extent / 128.0

        mt = MetaTensor(
            torch.from_numpy(data[np.newaxis]),
            affine=torch.from_numpy(affine.astype(np.float64)),
        )
        del data
        scaled_result = Spacing(pixdim=target_spacing, mode="bilinear")(mt)
        scaled_data = scaled_result.numpy()[0].astype(np.float32)
        scaled_affine = scaled_result.affine.numpy()
        del mt, scaled_result

        # Pad or crop to exactly 128³ — Spacing rounds voxel counts so the
        # result may be off by ±1 on any axis; ResizeWithPadOrCrop handles both.
        padded_data = np.asarray(
            ResizeWithPadOrCrop(TARGET_SHAPE)(scaled_data[np.newaxis]), dtype=np.float32
        )
        del scaled_data

        _save(padded_data[0], scaled_affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant E: %s", input_path.name, e)
