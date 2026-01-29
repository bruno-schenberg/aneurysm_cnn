import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_engine.src.nifti_resize import (
    resize_nifti, 
    resize_isotropic_with_padding, 
    center_crop_or_pad_nifti,
    resample_and_crop,
    resample_and_shrink
)

INPUT_NIFTI_PATH = Path("/mnt/data/cases-3/nifti")
OUTPUT_A_PATH = Path("/mnt/data/cases-3/dataset_A_resampled_cropped") # A - resample to 1mm + crop
OUTPUT_B_PATH = Path("/mnt/data/cases-3/dataset_B_resampled_shrunk") # B - resample to 1mm + shrink
OUTPUT_C_PATH = Path("/mnt/data/cases-3/dataset_C_cropped") # C - crop
OUTPUT_D_PATH = Path("/mnt/data/cases-3/dataset_D_shrunk") # D - shrink
OUTPUT_E_PATH = Path("/mnt/data/cases-3/dataset_E_isotropic_padded") # E - resample largest dimension 128px + pad
TARGET_SHAPE = (128, 128, 128)
MAX_WORKERS = os.cpu_count()  # Use all available CPU cores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(file_path: Path, output_paths: dict):
    """Processes a single NIfTI file with all 5 dataset generation methods."""
    try:
        filename = file_path.name
        logging.info(f"Processing {filename}...")

        # A - resample to 1mm + crop
        output_a_file = output_paths['A'] / filename
        resample_and_crop(str(file_path), str(output_a_file), target_shape=TARGET_SHAPE)

        # B - resample to 1mm + shrink
        output_b_file = output_paths['B'] / filename
        resample_and_shrink(str(file_path), str(output_b_file), target_shape=TARGET_SHAPE)

        # C - crop
        output_c_file = output_paths['C'] / filename
        center_crop_or_pad_nifti(str(file_path), str(output_c_file), target_shape=TARGET_SHAPE)

        # D - shrink
        output_d_file = output_paths['D'] / filename
        resize_nifti(str(file_path), str(output_d_file), target_shape=TARGET_SHAPE)

        # E - resample largest dimension 128px + pad
        output_e_file = output_paths['E'] / filename
        resize_isotropic_with_padding(str(file_path), str(output_e_file), target_size=TARGET_SHAPE[0])

        return f"Successfully processed {filename}"
    except Exception as e:
        return f"Failed to process {file_path.name}: {e}"

def main():
    """
    Main function to find and process all NIfTI files for resizing.
    """
    logging.info("Starting resizing process...")
    logging.info(f"Input directory: {INPUT_NIFTI_PATH}")
    logging.info(f"Dataset A Output: {OUTPUT_A_PATH}")
    logging.info(f"Dataset B Output: {OUTPUT_B_PATH}")
    logging.info(f"Dataset C Output: {OUTPUT_C_PATH}")
    logging.info(f"Dataset D Output: {OUTPUT_D_PATH}")
    logging.info(f"Dataset E Output: {OUTPUT_E_PATH}")

    # Find all NIfTI files, assuming a structure like '.../nifti/0/*.nii.gz'
    nifti_files = list(INPUT_NIFTI_PATH.rglob('*/*.nii.gz'))
    if not nifti_files:
        logging.warning("No .nii.gz files found in subdirectories. Checking top-level directory.")
        nifti_files = list(INPUT_NIFTI_PATH.glob('*.nii.gz'))

    if not nifti_files:
        logging.error("No .nii.gz files found to process. Exiting.")
        return

    logging.info(f"Found {len(nifti_files)} files to process.")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file_path in nifti_files:
            class_dir = file_path.parent.name
            
            # Create class-specific subdirectories for each dataset
            output_paths = {
                'A': OUTPUT_A_PATH / class_dir,
                'B': OUTPUT_B_PATH / class_dir,
                'C': OUTPUT_C_PATH / class_dir,
                'D': OUTPUT_D_PATH / class_dir,
                'E': OUTPUT_E_PATH / class_dir,
            }

            for path in output_paths.values():
                path.mkdir(parents=True, exist_ok=True)

            futures.append(executor.submit(process_file, file_path, output_paths))

        for future in as_completed(futures):
            logging.info(future.result())

    logging.info("Resizing process complete.")

if __name__ == "__main__":
    main()