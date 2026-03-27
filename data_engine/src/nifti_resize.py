import logging
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from monai.data import MetaTensor
from monai.transforms import Spacing, Resize, ResizeWithPadOrCrop, SpatialPad

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
    """Resample a volume to 1 mm isotropic voxel spacing.

    Uses MONAI Spacing (bilinear) via MetaTensor so the affine is correctly
    propagated through the resampling step. The caller does not need to update
    the affine manually.
    """
    # Wrap in MetaTensor so Spacing can read and update the affine
    mt = MetaTensor(
        torch.from_numpy(data[np.newaxis]),  # channel-first: (1, H, W, D)
        affine=torch.from_numpy(affine.astype(np.float64)),
    )
    transform = Spacing(pixdim=TARGET_SPACING_MM, mode="bilinear")
    result = transform(mt)
    resampled = result.numpy()[0].astype(np.float32)  # drop channel dim
    new_affine = result.affine.numpy()
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
        transform = ResizeWithPadOrCrop(TARGET_SHAPE)
        result = np.asarray(transform(data[np.newaxis]), dtype=np.float32)
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
        transform = Resize(TARGET_SHAPE, mode="trilinear", anti_aliasing=True)
        result = np.asarray(transform(data[np.newaxis]), dtype=np.float32)
        # Voxel size is scaled by old_shape / new_shape along each axis;
        # the affine columns (voxel-to-world vectors) are scaled accordingly.
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
        transform = ResizeWithPadOrCrop(TARGET_SHAPE)
        result = np.asarray(transform(data[np.newaxis]), dtype=np.float32)
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
        transform = Resize(TARGET_SHAPE, mode="trilinear", anti_aliasing=True)
        result = np.asarray(transform(data[np.newaxis]), dtype=np.float32)
        # The 1 mm affine is preserved as the nominal spacing for this variant;
        # the resize is treated as a fixed-grid resampling artefact, not a
        # physical scale change.
        _save(result[0], affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant B: %s", input_path.name, e)


def generate_variant_e(input_path: Path, output_path: Path) -> None:
    """Resample to isotropic at finest pitch → scale largest dim to 128 → zero-pad to 128³.

    Three-step pipeline that preserves aspect ratio while fitting every volume
    into a 128³ cube:

    Step 1 — Isotropic resampling: resample to isotropic voxels at the smallest
    original voxel pitch (finest resolution axis), removing anisotropy without
    sacrificing in-plane resolution.

    Step 2 — Uniform scale to 128: uniformly scale the isotropic volume so that
    its largest spatial dimension equals exactly 128 voxels, using anti-aliased
    trilinear interpolation to prevent aliasing during the typical downsampling
    case. Aspect ratio is preserved throughout.

    Step 3 — Symmetric zero-padding: pad the scaled volume to 128³ with zeros.
    Padding is always symmetric and never crops, so no brain tissue is removed
    in this step.
    """
    try:
        data, affine = _load(input_path)
        original_zooms = np.abs(np.diagonal(affine)[:3])
        min_zoom = float(np.min(original_zooms))

        # Step 1: Resample to isotropic at the smallest original voxel pitch so
        # that the finest-resolution axis is not degraded during isotropy correction.
        mt = MetaTensor(
            torch.from_numpy(data[np.newaxis]),
            affine=torch.from_numpy(affine.astype(np.float64)),
        )
        transform_spacing = Spacing(pixdim=min_zoom, mode="bilinear")
        iso_result = transform_spacing(mt)
        iso_data = iso_result.numpy()[0].astype(np.float32)
        iso_affine = iso_result.affine.numpy()
        iso_shape = np.array(iso_data.shape)  # (H, W, D)

        # Step 2: Uniform scale so max(shape) == 128; anti-aliasing prevents
        # aliasing artefacts when the isotropic volume is larger than 128³.
        scale = 128.0 / float(np.max(iso_shape))
        scaled_shape = tuple(max(1, round(float(s) * scale)) for s in iso_shape)
        transform_resize = Resize(scaled_shape, mode="trilinear", anti_aliasing=True)
        scaled_data = np.asarray(transform_resize(iso_data[np.newaxis]), dtype=np.float32)

        # Update affine: voxel size scales by iso_shape / scaled_shape.
        # Because the scale is uniform, each axis gets the same factor (1/scale).
        scale_per_axis = iso_shape / np.array(scaled_shape, dtype=float)
        scaled_affine = iso_affine.copy()
        scaled_affine[:3, :3] = iso_affine[:3, :3] * scale_per_axis[np.newaxis, :]

        # Step 3: Symmetric zero-pad to 128³ — brain fits inside, no cropping.
        transform_pad = SpatialPad(TARGET_SHAPE, method="symmetric")
        padded_data = np.asarray(transform_pad(scaled_data), dtype=np.float32)

        _save(padded_data[0], scaled_affine, output_path)
    except Exception as e:
        logger.error("Error processing %s for variant E: %s", input_path.name, e)
