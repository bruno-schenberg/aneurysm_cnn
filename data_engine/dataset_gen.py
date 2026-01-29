import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_engine.src.nifti_resize import resize_nifti, resize_isotropic_with_padding, center_crop_or_pad_nifti

INPUT_NIFTI_PATH = Path("/mnt/data/cases-3/nifti")
OUTPUT_RESIZED_PATH = Path("/mnt/data/cases-3/resized_shrunk") # For standard resize
OUTPUT_ISOTROPIC_PATH = Path("/mnt/data/cases-3/resized_isotropic_padded") # For isotropic resize
OUTPUT_CROPPED_PATH = Path("/mnt/data/cases-3/resized_center_cropped") # For center crop/pad
TARGET_SHAPE = (128, 128, 128)
MAX_WORKERS = os.cpu_count()  # Use all available CPU cores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(file_path: Path, output_resized_class_path: Path, output_isotropic_class_path: Path, output_cropped_class_path: Path):
    """Processes a single NIfTI file with both resizing methods."""
    try:
        filename = file_path.name
        logging.info(f"Processing {filename}...")

        # 1. Perform standard resize
        output_resized_file = output_resized_class_path / filename
        resize_nifti(str(file_path), str(output_resized_file), target_shape=TARGET_SHAPE)

        # 2. Perform isotropic resize with padding
        output_isotropic_file = output_isotropic_class_path / filename
        resize_isotropic_with_padding(str(file_path), str(output_isotropic_file), target_size=TARGET_SHAPE[0])

        # 3. Perform center crop with padding
        output_cropped_file = output_cropped_class_path / filename
        center_crop_or_pad_nifti(str(file_path), str(output_cropped_file), target_shape=TARGET_SHAPE)
        return f"Successfully processed {filename}"
    except Exception as e:
        return f"Failed to process {file_path.name}: {e}"

def main():
    """
    Main function to find and process all NIfTI files for resizing.
    """
    logging.info("Starting resizing process...")
    logging.info(f"Input directory: {INPUT_NIFTI_PATH}")
    logging.info(f"Output directory: {OUTPUT_RESIZED_PATH}")
    logging.info(f"Isotropic Output directory: {OUTPUT_ISOTROPIC_PATH}")
    logging.info(f"Cropped Output directory: {OUTPUT_CROPPED_PATH}")

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
            output_resized_class_path = OUTPUT_RESIZED_PATH / class_dir
            output_isotropic_class_path = OUTPUT_ISOTROPIC_PATH / class_dir
            output_cropped_class_path = OUTPUT_CROPPED_PATH / class_dir
            output_resized_class_path.mkdir(parents=True, exist_ok=True)
            output_isotropic_class_path.mkdir(parents=True, exist_ok=True)
            output_cropped_class_path.mkdir(parents=True, exist_ok=True)
            futures.append(executor.submit(process_file, file_path, output_resized_class_path, output_isotropic_class_path, output_cropped_class_path))

        for future in as_completed(futures):
            logging.info(future.result())

    logging.info("Resizing process complete.")

if __name__ == "__main__":
    main()