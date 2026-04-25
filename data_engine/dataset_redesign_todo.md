# Dataset Resize Redesign — To-Do

## Context

The current 4 dataset variants suffer from excessive cropping (A, C) and padding (B) because:
- Brain CTA data has a ~1.467:1 XY:Z physical FOV ratio (weighted avg across 739 exams)
- The 128x128x128 cube target mismatches this aspect ratio
- The 256x256x128 target (2:1) overshoots the ratio
- Variant C (native crop) is fundamentally broken at both sizes (85-99% crop)
- Variant A's 1mm fixed spacing is too fine for 128^3 (only 128mm FOV vs 180-240mm heads)
- Variant B's formula `max_extent / target_shape[0]` is wrong for non-cubic targets

## New 2x2 design

|                            | Native spacing | Isotropic resample |
|----------------------------|----------------|--------------------|
| **Center crop then shrink** | **C**          | **A**              |
| **Shrink only (no crop)**  | **D**          | **B**              |

Scientific questions:
1. **Rows** — Does focusing on central anatomy (removing peripheral skull/air/neck) help classification?
2. **Columns** — Does normalising voxel spacing across scanners help generalisation?

---

## 1. Target shapes

### 1.1 Small variant (replaces 128x128x128)
- [ ] Decide on target shape matching ~1.467:1 ratio (all dims divisible by 32)
  - Candidate: **96x96x64** (1.5:1, 0.6M voxels) — fast iteration, dev/debug
  - Candidate: **128x128x96** (1.33:1, 1.6M voxels) — close to current 128^3
  - Candidate: Other?
- [ ] Update `VALID_TARGET_SHAPES` in `nifti_resize.py`
- [ ] Update `OUTPUT_FOLDER_NAMES` in `dataset_gen.py`
- [ ] Decide fixed isotropic spacing for variant A at this size
- [ ] Update `TARGET_SPACING_MM_BY_SHAPE` in `nifti_resize.py`

### 1.2 Large variant (replaces 256x256x128)
- [ ] Decide on target shape matching ~1.467:1 ratio (all dims divisible by 32)
  - Candidate: **192x192x128** (1.5:1, 4.7M voxels, 44% smaller than 256x256x128)
  - Candidate: **256x256x176** (1.455:1, 11.5M voxels, closest ratio but 37% larger)
  - Candidate: **224x224x160** (1.4:1, 8.0M voxels)
- [ ] Update `VALID_TARGET_SHAPES` in `nifti_resize.py`
- [ ] Update `OUTPUT_FOLDER_NAMES` in `dataset_gen.py`
- [ ] Decide fixed isotropic spacing for variant A at this size
- [ ] Update `TARGET_SPACING_MM_BY_SHAPE` in `nifti_resize.py`

---

## 2. Variant A — isotropic resample + center crop (single pass)

Current: `Spacing(pixdim=fixed)` then `ResizeWithPadOrCrop(target)`.
New: same structure, but the fixed spacing is chosen so that the `ResizeWithPadOrCrop` step implicitly performs a center-crop that removes peripheral anatomy.

### Changes
- [ ] Choose fixed isotropic spacing per target shape that minimises weighted crop+pad
  - Spacing determines how much of the FOV survives into the target grid
  - Lower spacing = finer resolution = more voxels after resample = more cropping by `ResizeWithPadOrCrop`
  - Higher spacing = coarser = fewer voxels = more padding
  - Optimise for the dataset distribution: 322 BP002 (180mm), 218 BP042 (240mm), 183 BP082 (163-180mm), 16 BP240 (240mm)
- [ ] Update `TARGET_SPACING_MM_BY_SHAPE` with new values
- [ ] Verify with `check_resize_quality.py` that crop % is reasonable (target ~20% per axis, matching C's 80% center crop intent)
- [ ] Update docstrings in `_variant_a_from_data` to reflect new strategy description
- [ ] Update module-level docstring table in `nifti_resize.py`

### No code logic changes needed
The implementation (`Spacing` then `ResizeWithPadOrCrop`) stays the same — only the spacing constant changes. The crop/pad behaviour emerges from the spacing choice.

---

## 3. Variant B — isotropic resample, never crop (formula fix)

Current: `target_spacing = max_physical_extent / target_shape[0]` — broken for non-cubic targets.
New: `target_spacing = max_i(physical_extent_i / target_shape_i)` — correct generalisation.

### Changes
- [ ] Fix spacing computation in `_variant_b_from_data`:
  ```python
  # old (wrong for non-cubic):
  target_spacing = max_physical_extent / float(target_shape[0])

  # new (correct):
  original_zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
  extents = np.array(data.shape, dtype=float) * original_zooms
  ratios = extents / np.array(target_shape, dtype=float)
  target_spacing = float(np.max(ratios))
  ```
- [ ] Verify: after this fix, B should have 0% crop for all exams at all target shapes
- [ ] Update docstrings in `_variant_b_from_data`
- [ ] Update module-level docstring (B description)

---

## 4. Variant C — native resolution, 80% center crop then shrink

Current: `ResizeWithPadOrCrop(target)` — catastrophic 85-99% crop at native resolution.
New: center-crop 80% of each axis in voxel space, then `Resize(target)` to shrink.

### Changes
- [ ] Rewrite `_variant_c_from_data`:
  1. Compute 80% crop size per axis: `crop_shape = tuple(int(s * 0.8) for s in data.shape)`
  2. Apply center crop: `ResizeWithPadOrCrop(crop_shape)` (lossless, no interpolation)
  3. Shrink to target: `Resize(target_shape, mode="trilinear")`
- [ ] Update affine: scale columns by `crop_shape / target_shape` to reflect new voxel sizes
- [ ] Update docstrings
- [ ] Update module-level docstring table

### Interpolation count: 1 (the Resize step only; the center crop is lossless array slicing)

---

## 5. Variant D — native resolution, shrink only

Current: `Resize(target_shape, mode="trilinear")` — works correctly, 0% crop, 0% pad.
New: no logic changes, only target shape changes.

### Changes
- [ ] No code changes to `_variant_d_from_data` (just new target shapes)
- [ ] Update docstrings if the module-level table changes
- [ ] Verify 0% crop / 0% pad at new target shapes

---

## 6. Supporting changes

- [ ] Update `check_resize_quality.py` to work with new target shapes
- [ ] Run quality check on 4 sample exams at both new target shapes, all 4 variants
- [ ] Update tests in `test_nifti_resize.py`:
  - [ ] Update test for variant A (spacing assertion will change)
  - [ ] Update test for variant B (formula changed)
  - [ ] Update test for variant C (new crop+shrink logic)
  - [ ] Add tests for new target shapes
- [ ] Update `CLAUDE.md` dataset variant table (A/B/C/D descriptions)
- [ ] Update `experiments.json` / experiment configs for new dataset paths
- [ ] Remove old `VALID_TARGET_SHAPES` entries if old sizes are fully retired
- [ ] Regenerate datasets on SSD after all code changes verified

---

## 7. HU windowing — training pipeline (training_engine/src/data_preprocess.py)

Current `ScaleIntensityd` normalises to [0, 1] using the **per-volume min and max**, which
is driven by air (−1000 HU) and bone (700–3000 HU). Vessels (200–400 HU) and brain
parenchyma (20–40 HU) get compressed into a narrow band that shifts per patient —
the model cannot learn consistent tissue representations across scanners.

### Changes
- [ ] Replace `ScaleIntensityd` with `ScaleIntensityRanged` in `get_transforms()`:
  ```python
  # Remove:
  ScaleIntensityd(keys=keys),

  # Add:
  ScaleIntensityRanged(keys=keys, a_min=0, a_max=700, b_min=0.0, b_max=1.0, clip=True),
  ```
- [ ] Add `ScaleIntensityRanged` to the import block in `data_preprocess.py`
- [ ] Decide on the HU window — `0–700` captures parenchyma through contrast-enhanced
  vessels without bone; alternatives to consider:
  - `0–500` — tighter, less bone bleed-through
  - `-100–800` — wider, includes some bone context
  - Treat window as an experiment config parameter (`HU_MIN`, `HU_MAX` in `experiments.json`)
- [ ] Update `get_transforms` docstring to describe `ScaleIntensityRanged` behaviour
- [ ] Update `build_dataloaders` docstring (references intensity scaling)
- [ ] Verify with a sample volume that vessel intensities map to a visible [0,1] range
  after windowing (e.g. contrast-enhanced vessels should land around 0.3–0.6)

### Note on CacheDataset interaction
`ScaleIntensityRanged` is deterministic, so `CacheDataset` will correctly cache it
alongside `LoadImaged`, `EnsureChannelFirstd`, and `Resized`. No changes needed to
the caching logic.

---

## 8. Validation checklist (per target shape, per variant)

For each of the 4 sample exams (BP002, BP042, BP082, BP240):
- [ ] Output shape matches target exactly
- [ ] Crop % is within acceptable range (A: ~20% matching C, B: 0%, C: 20%, D: 0%)
- [ ] Pad % is within acceptable range (A: minimal, B: <25%, C: 0%, D: 0%)
- [ ] No upscaling occurs (resampled/cropped volume >= target in all dims)
- [ ] Affine matrix correctly reflects output voxel spacing
- [ ] Visual inspection: anatomy is centred, no severe distortion
