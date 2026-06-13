# Normalization: Min-Max vs Z-Score

## Current approach
`ScaleIntensityd` — min-max normalization, scales to [0, 1].

## Alternative: z-score
`NormalizeIntensityd` — subtracts mean, divides by std → zero mean, unit variance.

### Why z-score may be better
- Robust to outlier voxels (a single bright artifact doesn't compress the whole range)
- Better cross-scanner consistency (different scanners produce different intensity ranges)
- Standard in medical imaging DL (mdbrain uses it, most papers use it)

### How to swap (data_preprocess.py)
```python
# current
ScaleIntensityd(keys=keys)           # min-max → [0, 1]

# z-score alternative
NormalizeIntensityd(keys=keys)       # (x - mean) / std
```

Both are MONAI transforms, one-line change. Add `NormalizeIntensityd` to the import.

### Note on augmentation interaction
`RandScaleIntensityd` and `RandGaussianNoised` parameters were tuned for [0, 1] range.
After switching to z-score, the noise std and scale factors may need adjusting since
the value range changes (z-scored data typically spans roughly [-2, +5] for TOF-MRA
depending on the anatomy).

## TODO
- Run one experiment with each normalization and compare
- Check if augmentation params need retuning after switching
