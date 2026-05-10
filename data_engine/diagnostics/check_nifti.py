"""
check_nifti.py

Spot-checks a NIfTI file: shape, dtype, and value range.
Usage: python data_engine/check_nifti.py <path_to_nifti>
"""

import sys
import nibabel as nib
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/cases-3/dataset_D_shrunk/1/BP001.nii.gz"

img = nib.load(path)
d = img.get_fdata()

print(f"File:    {path}")
print(f"Shape:   {img.shape}")
print(f"Dtype:   {d.dtype}")
print(f"Min:     {d.min():.1f}")
print(f"Max:     {d.max():.1f}")
print(f"Mean:    {d.mean():.1f}")
print(f"Non-zero voxels: {np.count_nonzero(d)} / {d.size} ({100*np.count_nonzero(d)/d.size:.1f}%)")
