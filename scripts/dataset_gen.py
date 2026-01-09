import os
import glob
from resizing.nifti_resize import resize_nifti

INPUT_NIFTI_PATH = "/mnt/data/cases-3/nifti"
OUTPUT_RESIZED_PATH = "/mnt/data/cases-3/resized_128"


def main():
    """
    Main function to find and process all NIfTI files for resizing.
    """
    print("Starting resizing process...")
    print(f"Input directory: {INPUT_NIFTI_PATH}")
    print(f"Output directory: {OUTPUT_RESIZED_PATH}")

    # Iterate through the class subdirectories ('0', '1', etc.)
    for class_dir in os.listdir(INPUT_NIFTI_PATH):
        input_class_path = os.path.join(INPUT_NIFTI_PATH, class_dir)

        if not os.path.isdir(input_class_path):
            continue

        print(f"\nProcessing class directory: {class_dir}")

        # Create the corresponding output directory
        output_class_path = os.path.join(OUTPUT_RESIZED_PATH, class_dir)
        os.makedirs(output_class_path, exist_ok=True)

        # Find all NIfTI files in the input class directory
        nifti_files = glob.glob(os.path.join(input_class_path, '*.nii.gz'))

        if not nifti_files:
            print("  - No .nii.gz files found.")
            continue

        total_files = len(nifti_files)
        print(f"  - Found {total_files} files to process.")

        # Process each file
        for i, file_path in enumerate(nifti_files):
            filename = os.path.basename(file_path)
            output_file_path = os.path.join(output_class_path, filename)
            print(f"  ({i+1}/{total_files}) Processing {filename}...")
            resize_nifti(file_path, output_file_path)

    print("\nResizing process complete.")


if __name__ == "__main__":
    # Before running, ensure you have the necessary libraries installed:
    # pip install numpy nibabel scipy
    main()