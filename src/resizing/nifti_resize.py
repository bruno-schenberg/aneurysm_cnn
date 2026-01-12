import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img

TARGET_SHAPE = np.array([128, 128, 128])

def resize_nifti(input_path, output_path):
    """
    Loads a NIfTI file, resizes it to a fixed size of 128x128x128 using
    trilinear interpolation, and saves it.

    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the resized NIfTI file.
    """
    try:
        img = nib.load(input_path)
        data = img.get_fdata()
        affine = img.affine

        # 2. Get current shape
        current_shape = np.array(data.shape)

        # 2a. Check if resampling is necessary
        if np.array_equal(current_shape, TARGET_SHAPE):
            print(f"  - Image shape is already {TARGET_SHAPE}. Copying file directly.")
            nib.save(img, output_path)  # Just save the original image
            return

        # 3. Calculate the zoom factor needed to reach the target shape
        # This is a simple, non-isotropic scaling.
        zoom_factor = TARGET_SHAPE / np.array(current_shape, dtype=float)

        # 4. Use nilearn to resample. It correctly handles the affine transformation.
        # Create a new target affine that reflects the new voxel sizes.
        new_affine = nib.affines.rescale_affine(affine, current_shape, zoom_factor)
        resized_img = resample_img(img, target_affine=new_affine, target_shape=TARGET_SHAPE, interpolation='continuous')

        # 7. Save the resampled image
        nib.save(resized_img, output_path)
        print(f"  - Resized and saved to {output_path}")

    except Exception as e:
        print(f"  - ERROR processing {os.path.basename(input_path)}: {e}")

def resize_isotropic_with_padding(input_path, output_path, target_size=128):
    """
    Resamples a NIfTI file to be isotropic with voxels of the smallest original
    dimension, then resizes and pads it into a cubic shape (target_size^3)
    using nilearn for robust affine handling.

    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the processed NIfTI file.
        target_size (int): The edge length of the final cubic image.
    """
    final_shape = (target_size, target_size, target_size)
    try:
        # 1. Load the NIfTI image
        img = nib.load(input_path)

        # 2. Determine target isotropic voxel size (zooms)
        # We use the smallest of the original voxel dimensions to avoid losing data.
        original_zooms = np.array(img.header.get_zooms()[:3])
        target_zoom = min(original_zooms)

        # 3. Calculate the new affine matrix for the isotropic resampling.
        # nilearn will use this to determine how to resample the data.
        # We scale the rotational/scaling part of the affine by the zoom factor.
        iso_affine = img.affine.copy()
        np.fill_diagonal(iso_affine, [target_zoom, target_zoom, target_zoom, 1])

        # 4. Use nilearn.image.resample_img to perform all steps at once:
        #   - Resample to isotropic voxels defined by iso_affine.
        #   - Rescale and pad/crop to fit the final_shape.
        # 'continuous' = trilinear interpolation. 'blackman' is a higher quality alternative.
        resized_img = resample_img(img, target_affine=iso_affine, target_shape=final_shape, interpolation='continuous')
        nib.save(resized_img, output_path)
        print(f"  - Isotopically resized and saved to {output_path}")

    except Exception as e:
        print(f"  - ERROR processing {os.path.basename(input_path)}: {e}")