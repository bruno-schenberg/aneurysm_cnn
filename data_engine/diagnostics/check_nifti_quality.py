import os
import nibabel as nib
import numpy as np
import pydicom
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

NIFTI_DIR = "/mnt/data/cases-3/nifti"


def check_nifti_file(nifti_path: str) -> None:
    print(f"\n=== Checking: {nifti_path} ===")

    print(f"\n1. Loading NIfTI: {nifti_path}")
    img = nib.load(nifti_path)
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

    if MATPLOTLIB_AVAILABLE:
        print("\n2. Generating a 2D slice plot...")
        z_mid = data.shape[2] // 2
        slice_2d = data[:, :, z_mid]

        plt.figure(figsize=(6, 6))
        plt.imshow(slice_2d.T, cmap='gray', origin='lower')
        name = os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0]
        plt.title(f"{name} — Middle Slice (Z={z_mid})")
        plt.axis('off')

        plot_path = f"/tmp/nifti_slice_{name}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved middle slice plot to: {plot_path}")
    else:
        print("\nMatplotlib not installed, skipping plot generation.")


def main():
    # Collect one sample from each class subdirectory
    samples = []
    for class_label in ("0", "1"):
        class_dir = os.path.join(NIFTI_DIR, class_label)
        if not os.path.isdir(class_dir):
            print(f"Class directory not found: {class_dir}")
            continue
        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".nii.gz"))
        if files:
            samples.append(os.path.join(class_dir, files[0]))

    if not samples:
        print(f"No NIfTI files found in {NIFTI_DIR}")
        return

    for path in samples:
        check_nifti_file(path)


if __name__ == "__main__":
    main()
