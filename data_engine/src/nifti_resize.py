import os
import logging
import numpy as np
import nibabel as nib
from nilearn.image import resample_img, resample_to_img

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_nifti(input_path: str, output_path: str, target_shape: tuple):
    """
    Loads a NIfTI file, resizes it to a fixed size of 128x128x128 using
    trilinear interpolation, and saves it.

    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the resized NIfTI file.
        target_shape (tuple or list): The target shape for the output image (e.g., (128, 128, 128)).
    """
    try:
        img = nib.load(input_path, mmap=False) # mmap=False to avoid issues with file handles
        data = img.get_fdata()
        affine = img.affine

        # 2. Get current shape
        current_shape = np.array(data.shape)
 
        # 2a. Check if resampling is necessary
        if np.array_equal(current_shape, target_shape):
            logging.info(f"Image shape is already {target_shape}. Copying {os.path.basename(input_path)} directly.")
            nib.save(img, output_path)  # Just save the original image
            return

        # 3. Calculate the zoom factor needed to reach the target shape
        # This is a simple, non-isotropic scaling.
        zoom_factor = np.array(target_shape) / np.array(current_shape, dtype=float)

        # 4. Use nilearn to resample. It correctly handles the affine transformation.
        # Create a new target affine that reflects the new voxel sizes.
        new_affine = nib.affines.rescale_affine(affine, current_shape, zoom_factor)
        resized_img = resample_img(img, target_affine=new_affine, target_shape=target_shape, interpolation='continuous')

        # 7. Save the resampled image
        resized_img.header.set_zooms(np.abs(np.diagonal(resized_img.affine)[:3]))
        nib.save(resized_img, output_path)
        logging.info(f"Resized {os.path.basename(input_path)} and saved to {output_path}")

    except (nib.filebasedimages.ImageFileError, FileNotFoundError) as e:
        logging.error(f"ERROR loading or finding file {os.path.basename(input_path)}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {os.path.basename(input_path)}: {e}")

def resize_isotropic_with_padding(input_path: str, output_path: str, target_size: int = 128, fill_value: float = 0.0):
    """
    Resamples a NIfTI file to be isotropic with voxels of the smallest original
    dimension, then resizes and pads it into a cubic shape (target_size^3)
    using nilearn for robust affine handling.

    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the processed NIfTI file.
        target_size (int): The edge length of the final cubic image.
        fill_value (float): The value to use for padding.
    """
    final_shape = (target_size, target_size, target_size)
    try:
        # 1. Load the NIfTI image
        img = nib.load(input_path, mmap=False)

        # 2. Make the image isotropic by resampling to the smallest voxel size
        original_zooms = np.array(img.header.get_zooms()[:3])
        target_zoom = min(original_zooms)
        
        # Create a new affine for the isotropic image
        iso_affine = np.diag([target_zoom, target_zoom, target_zoom, 1.0])
        iso_img = resample_to_img(img, target_affine=iso_affine, interpolation='continuous')

        # 3. Create a target canvas image with the final desired shape and affine
        # This ensures the final image is centered correctly.
        target_affine = iso_affine.copy()
        target_img = nib.Nifti1Image(np.full(final_shape, fill_value), affine=target_affine)

        # 4. Resample the isotropic image to the target canvas.
        # This will pad/crop the image to the final_shape.
        resized_img = resample_to_img(iso_img, target_img, interpolation='continuous', fill_value=fill_value)

        nib.save(resized_img, output_path)
        logging.info(f"Isotropically resized {os.path.basename(input_path)} and saved to {output_path}")

    except (nib.filebasedimages.ImageFileError, FileNotFoundError) as e:
        logging.error(f"ERROR loading or finding file {os.path.basename(input_path)}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {os.path.basename(input_path)}: {e}")

def center_crop_or_pad_nifti(input_path: str, output_path: str, target_shape: tuple, fill_value: float = 0.0):
    """
    Crops or pads a NIfTI image to a target shape from the center.

    If the image is larger than the target shape, it's cropped. If it's smaller,
    it's padded with `fill_value`. The affine matrix is adjusted to reflect the
    new origin.

    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the processed NIfTI file.
        target_shape (tuple): The target shape (e.g., (128, 128, 128)).
        fill_value (float): The value to use for padding if needed.
    """
    try:
        img = nib.load(input_path, mmap=False)
        data = img.get_fdata()
        affine = img.affine
        current_shape = np.array(data.shape)
        target_shape = np.array(target_shape)

        # Calculate the difference in shape
        diff = target_shape - current_shape

        # Calculate crop/pad amounts for each dimension
        crop_before = np.maximum(0, -diff) // 2
        crop_after = np.maximum(0, -diff) - crop_before
        pad_before = np.maximum(0, diff) // 2
        pad_after = np.maximum(0, diff) - pad_before

        # Define slices for cropping from the original data
        slices = tuple(slice(cb, s - ca) for cb, s, ca in zip(crop_before, current_shape, crop_after))
        cropped_data = data[slices]

        # Define padding widths
        pad_widths = tuple((pb, pa) for pb, pa in zip(pad_before, pad_after))
        padded_data = np.pad(cropped_data, pad_widths, mode='constant', constant_values=fill_value)

        # Adjust the affine matrix for the new origin
        translation = nib.affines.from_translation(crop_before - pad_before)
        new_affine = np.dot(affine, translation)

        new_img = nib.Nifti1Image(padded_data, new_affine, img.header)
        nib.save(new_img, output_path)
        logging.info(f"Center-cropped/padded {os.path.basename(input_path)} and saved to {output_path}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during cropping of {os.path.basename(input_path)}: {e}")

def resample_to_1mm_isotropic_and_pad(input_path: str, output_path: str, target_shape: tuple, fill_value: float = 0.0):
    """
    Resamples a NIfTI file to a fixed 1mm x 1mm x 1mm isotropic voxel size,
    then pads or crops it to a target cubic shape while maintaining orientation.

    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the processed NIfTI file.
        target_shape (tuple): The target shape for the output image (e.g., (128, 128, 128)).
        fill_value (float): The value to use for padding.
    """
    try:
        # 1. Load the NIfTI image
        img = nib.load(input_path, mmap=False)

        # 2. Create a target affine with 1mm isotropic voxels.
        # We copy the original affine to preserve the orientation and origin,
        # then set the voxel sizes (the diagonal elements) to 1.0.
        target_affine = img.affine.copy()
        np.fill_diagonal(target_affine, [1.0, 1.0, 1.0, 1.0])

        # 3. Create a target "canvas" image with the final desired shape and the 1mm affine.
        # This ensures the final image is centered correctly within the 128^3 volume.
        target_img = nib.Nifti1Image(np.full(target_shape, fill_value), affine=target_affine)

        # 4. Resample the original image directly to the target canvas.
        # nilearn handles the two steps (resampling to 1mm and padding/cropping) in one go.
        resampled_img = resample_to_img(img, target_img, interpolation='continuous', fill_value=fill_value)

        # 5. Save the final image
        nib.save(resampled_img, output_path)
        logging.info(f"Resampled to 1mm isotropic {os.path.basename(input_path)} and saved to {output_path}")

    except (nib.filebasedimages.ImageFileError, FileNotFoundError) as e:
        logging.error(f"ERROR loading or finding file {os.path.basename(input_path)}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during 1mm resampling of {os.path.basename(input_path)}: {e}")