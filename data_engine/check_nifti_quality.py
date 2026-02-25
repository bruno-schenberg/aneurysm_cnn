import os
import nibabel as nib
import numpy as np
import pydicom
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
from src.nifti_utils import convert_series_to_nifti

SAMPLE_DICOM_DIR = "/home/aneurysm_cnn/data_engine/dataset/sample-raw/BP05- 2838474/Anonymous_Mr_712482402_20170105"
OUTPUT_NIFTI = "/tmp/sample_output.nii.gz"

def main():
    print("1. Converting DICOM to NIfTI...")
    success = convert_series_to_nifti(SAMPLE_DICOM_DIR, OUTPUT_NIFTI)
    if not success:
        print("Conversion failed!")
        return

    print(f"\n2. Loading generated NIfTI: {OUTPUT_NIFTI}")
    img = nib.load(OUTPUT_NIFTI)
    data = img.get_fdata()
    header = img.header

    print("\n--- NIfTI Header Info ---")
    print(f"Shape (Dimensions): {data.shape}")
    print(f"Voxel Spacing (Zooms): {header.get_zooms()}")
    print(f"Data Type: {header.get_data_dtype()}")
    
    print("\n--- Affine Matrix (Orientation & Spacing) ---")
    print(img.affine)

    print("\n--- Image Data Quality Checks ---")
    print(f"Min Intensity: {np.min(data)}")
    print(f"Max Intensity: {np.max(data)}")
    print(f"Mean Intensity: {np.mean(data):.2f}")
    print(f"NaN Count: {np.isnan(data).sum()}")
    print(f"Inf Count: {np.isinf(data).sum()}")

    # Compare with a single DICOM file
    dcm_files = [f for f in os.listdir(SAMPLE_DICOM_DIR) if f.endswith('.dcm')]
    if dcm_files:
        dcm_path = os.path.join(SAMPLE_DICOM_DIR, dcm_files[0])
        ds = pydicom.dcmread(dcm_path)
        print("\n--- Original DICOM Metadata (Sample) ---")
        print(f"Pixel Spacing: {getattr(ds, 'PixelSpacing', 'N/A')}")
        print(f"Slice Thickness: {getattr(ds, 'SliceThickness', 'N/A')}")
        print(f"Rows x Columns: {getattr(ds, 'Rows', 'N/A')} x {getattr(ds, 'Columns', 'N/A')}")

    if MATPLOTLIB_AVAILABLE:
        print("\n3. Generating a 2D slice plot...")
        # Get the middle slice along the Z axis (usually the last dimension)
        z_mid = data.shape[2] // 2
        slice_2d = data[:, :, z_mid]
        
        plt.figure(figsize=(6, 6))
        # NIfTI data often needs transposing or rotating for matplotlib depending on orientation
        plt.imshow(slice_2d.T, cmap='gray', origin='lower')
        plt.title(f"Middle Slice (Z={z_mid})")
        plt.axis('off')
        
        plot_path = "/tmp/nifti_slice.png"
        plt.savefig(plot_path)
        print(f"Saved middle slice plot to: {plot_path}")
    else:
        print("\nMatplotlib not installed, skipping plot generation.")

if __name__ == "__main__":
    main()
