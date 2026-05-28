## nifti_resize.py

**Location:** `data_engine/src/nifti_resize.py`

**Purpose:** Generates the six preprocessing variants (A–F) from clean NIfTI volumes produced by the DICOM conversion step, implementing a 3x2 factorial design of three resampling strategies crossed with two centre-crop strategies. Called by `dataset_gen.py`, one call per file per variant.

---

### Module-level constants

| Name | Value / Type | Purpose |
|---|---|---|
| `TARGET_SHAPE` | `(128, 128, 128)` | Legacy default output grid; kept for backwards compatibility only. |
| `CROP_FRACTION` | `0.8` (`float`) | Fraction of each native axis retained by the centre-crop step (variants A, C, E). |
| `VALID_TARGET_SHAPES` | `Dict[str, Tuple[int,int,int]]` | Registry of all supported output grids; adding a new resolution requires only a matching entry here and in `dataset_gen.py`. |
| `E_SPACING_MM_BY_SHAPE` | `Dict[Tuple[int,int,int], float]` | Fixed isotropic mm/voxel for variant E per target shape. Derived as `192 mm / target_XY` (192 mm = 240 mm × 0.8, the largest 80%-cropped XY FOV in the dataset). |
| `F_SPACING_MM_BY_SHAPE` | `Dict[Tuple[int,int,int], float]` | Fixed isotropic mm/voxel for variant F per target shape. Derived as `240 mm / target_XY` (240 mm = largest full XY FOV in the dataset). |

---

### Private helpers

#### `_load`

**What it does:** Loads a NIfTI file from disk and returns the voxel array and affine matrix.

**Design rationale:** `mmap=False` forces NiBabel to read the entire file into RAM immediately, avoiding shared file-handle problems when called from parallel worker processes. `get_fdata(dtype=np.float32)` casts on load rather than defaulting to float64, halving memory usage for large volumes.

**Parameters:**
- `path` (`Path`) — absolute path to a `.nii.gz` file.

**Returns:** `tuple[np.ndarray, np.ndarray]` — `(data, affine)` where `data` has shape `(H, W, D)` and dtype `float32`, and `affine` has shape `(4, 4)`.

**Side effects / exceptions:** Raises whatever NiBabel raises on a missing or corrupt file (e.g., `FileNotFoundError`, `nib.filebasedimages.ImageFileError`). These propagate to the public `generate_variant_*` wrappers, which catch all exceptions and log them.

---

#### `_save`

**What it does:** Asserts that the output array has the expected shape, then writes it as a compressed NIfTI file.

**Design rationale:** The `assert` statement acts as a cheap contract check: every variant pipeline must produce exactly `target_shape` before writing. This surfaces shape bugs immediately with a descriptive message rather than silently writing a wrongly-sized file that would corrupt dataset batches downstream.

**Parameters:**
- `data` (`np.ndarray`) — volume to save; must satisfy `data.shape == target_shape`.
- `affine` (`np.ndarray`) — `(4, 4)` affine matrix embedded in the NIfTI header.
- `path` (`Path`) — output file path (expected to end in `.nii.gz`).
- `target_shape` (`Tuple[int, int, int]`, default `TARGET_SHAPE`) — expected shape; used only for the assertion.

**Returns:** `None`.

**Side effects:** Writes a `.nii.gz` file to disk. Raises `AssertionError` if `data.shape != target_shape`. Raises OS-level errors if the output directory does not exist or is not writable.

---

#### `_center_crop`

**What it does:** Crops the central `crop_fraction` of each spatial axis via array slicing, with no interpolation.

**Design rationale:** Uses MONAI's `ResizeWithPadOrCrop` purely for its centred-slicing logic. Because the target is smaller than the input, MONAI performs only a slice operation (no interpolation), making the crop genuinely lossless. A temporary channel dimension (`data[np.newaxis]`) is added to satisfy MONAI's channel-first convention and stripped on return.

**Parameters:**
- `data` (`np.ndarray`) — input volume with shape `(H, W, D)`.
- `crop_fraction` (`float`, default `CROP_FRACTION = 0.8`) — fraction of each axis to retain.

**Returns:** `np.ndarray` — cropped volume with shape `(int(H*f), int(W*f), int(D*f))`, dtype `float32`.

**Side effects / exceptions:** None. The integer truncation in `int(s * crop_fraction)` means the actual retained fraction can be very slightly less than `crop_fraction` for odd-sized axes (off-by-one at most).

---

#### `_dynamic_spacing`

**What it does:** Computes the single isotropic voxel spacing (in mm) that maps the largest physical extent of the volume to the corresponding target dimension without cropping any axis.

**Design rationale:** The formula `spacing = max_i(physical_extent_i / target_i)` guarantees that after resampling, the volume fits within `target_shape` on every axis simultaneously. Axes with smaller physical extents land proportionally below their target dimension and are zero-padded later. This is a per-exam, per-call computation, so it adapts to whatever volume (full or cropped) is passed in.

**Parameters:**
- `data` (`np.ndarray`) — volume whose shape (in voxels) is used; `(H, W, D)`.
- `affine` (`np.ndarray`) — `(4, 4)` affine; column norms of the upper-left 3x3 block give the per-axis voxel sizes in mm.
- `target_shape` (`Tuple[int, int, int]`) — target output grid in voxels.

**Returns:** `float` — isotropic spacing in mm.

**Side effects / exceptions:** None. Note that voxel sizes are extracted as column norms of `affine[:3, :3]` (`np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))`), which is correct for standard NIfTI affines that encode spacing as column-vector lengths but could be wrong for affines with a shear component.

---

#### `_resample_isotropic`

**What it does:** Resamples a volume to a uniform isotropic voxel spacing using bilinear interpolation, returning the resampled array and its updated affine.

**Design rationale:** Wraps the input in a MONAI `MetaTensor` (PyTorch tensor + embedded affine) to drive MONAI's `Spacing` transform, which handles the physical-space coordinate mapping internally. The affine is passed as `float64` to match MONAI's internal precision requirements, then the result is cast back to `float32`. Explicit `del` calls on `mt` and `result` free the GPU/CPU tensors immediately, keeping memory pressure low in parallel workers.

**Resize / resampling strategy — library coupling:** This function is the primary resampling site and is **tightly coupled to MONAI**. It depends on `monai.transforms.Spacing` and `monai.data.MetaTensor`. Making it library-agnostic would require replacing these with, e.g., `scipy.ndimage.zoom` or `SimpleITK.Resample`, and manually computing the new voxel grid size and updated affine.

**Parameters:**
- `data` (`np.ndarray`) — input volume `(H, W, D)`, dtype `float32`.
- `affine` (`np.ndarray`) — `(4, 4)` current affine matrix.
- `spacing` (`float`) — target isotropic voxel spacing in mm.

**Returns:** `tuple[np.ndarray, np.ndarray]` — `(resampled_data, new_affine)` where `resampled_data` has dtype `float32` and a new shape determined by MONAI based on the spacing ratio, and `new_affine` is the updated `(4, 4)` matrix.

**Side effects / exceptions:** Allocates a PyTorch tensor. May raise if MONAI or PyTorch is unavailable or if the affine is malformed.

---

### Per-variant transform helpers (`_variant_*_from_data`)

These six private functions implement the transform logic for each variant. They receive a pre-loaded `(data, affine)` pair so the public wrappers can load a file once and pass the arrays to multiple variants without redundant I/O. None return a value; all write their result via `_save`.

**Common signature:**

```python
def _variant_X_from_data(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None
```

**Parameters (all variants):**
- `data` (`np.ndarray`) — pre-loaded volume `(H, W, D)`.
- `affine` (`np.ndarray`) — `(4, 4)` affine from the source NIfTI.
- `output_path` (`Path`) — destination `.nii.gz` path.
- `target_shape` (`Tuple[int, int, int]`, default `TARGET_SHAPE`) — output voxel grid.

**Returns:** `None` (all variants). Side effect: writes one `.nii.gz` file.

#### `_variant_a_from_data`

**What it does:** 80% centre crop → dynamic isotropic resample (spacing computed on the cropped volume) → zero-pad to `target_shape`.

**Design rationale:** Computing `_dynamic_spacing` on the already-cropped volume ensures the spacing guarantees no-crop relative to the cropped extent, not the full volume. This is A's distinguishing property versus B.

**Pipeline:** `_center_crop` → `_dynamic_spacing` → `_resample_isotropic` → `ResizeWithPadOrCrop` (zero-pad only) → `_save`.

---

#### `_variant_b_from_data`

**What it does:** Dynamic isotropic resample of the full native volume → zero-pad to `target_shape`.

**Design rationale:** Structurally identical to A but omits the centre crop. The spacing is computed on the full volume extent, so peripheral anatomy is preserved at the cost of a slightly coarser voxel size.

**Pipeline:** `_dynamic_spacing` → `_resample_isotropic` → `ResizeWithPadOrCrop` → `_save`.

---

#### `_variant_c_from_data`

**What it does:** 80% centre crop (lossless) → trilinear resize directly to `target_shape`, with affine scaled to reflect the new effective voxel size.

**Design rationale:** Bypasses physical-space resampling entirely; the resize is purely geometric. This is the simplest crop-variant pipeline: one interpolation pass via MONAI `Resize`. The affine is updated by column-scaling `affine[:3, :3]` by `crop_shape / target_shape` to keep the physical coordinates consistent.

**Flag — potential simplification:** `crop_shape` is computed twice: once inside `_center_crop` (implicitly) and once explicitly in `_variant_c_from_data` for the affine scaling. Extracting `crop_shape` once before calling `_center_crop` and passing it to both would remove the duplication.

**Pipeline:** `_center_crop` → `Resize(mode="trilinear")` → affine column-scale → `_save`.

---

#### `_variant_d_from_data`

**What it does:** Trilinear resize of the full native volume directly to `target_shape`, with affine scaled accordingly.

**Design rationale:** The simplest possible pipeline — no crop, no physical-space resampling. Serves as the no-preprocessing baseline in the factorial design. Affine is updated identically to C using `old_shape / target_shape` as the scale factor.

**Pipeline:** `Resize(mode="trilinear")` → affine column-scale → `_save`.

---

#### `_variant_e_from_data`

**What it does:** 80% centre crop → fixed isotropic resample at `E_SPACING_MM_BY_SHAPE[target_shape]` → zero-pad to `target_shape`.

**Design rationale:** Uses a dataset-wide fixed spacing (not per-exam) derived from the largest plausible 80%-cropped FOV (192 mm), guaranteeing no exam will ever be cropped during the pad step. This makes variant E scanner-independent and deterministic across exams, unlike A.

**Side effects / exceptions:** Raises `KeyError` if `target_shape` is not in `E_SPACING_MM_BY_SHAPE` (i.e., an unregistered output grid is used).

**Pipeline:** `_center_crop` → `_resample_isotropic(fixed spacing)` → `ResizeWithPadOrCrop` → `_save`.

---

#### `_variant_f_from_data`

**What it does:** Fixed isotropic resample of the full native volume at `F_SPACING_MM_BY_SHAPE[target_shape]` → zero-pad to `target_shape`.

**Design rationale:** The most physically principled variant: every voxel in every output volume represents the same physical volume of tissue regardless of the source scanner. The fixed spacing is derived from the largest full-volume FOV in the dataset (240 mm), so no exam is ever cropped. Equivalent to E but applied to the full (uncropped) volume.

**Side effects / exceptions:** Raises `KeyError` if `target_shape` is not in `F_SPACING_MM_BY_SHAPE`.

**Pipeline:** `_resample_isotropic(fixed spacing)` → `ResizeWithPadOrCrop` → `_save`.

---

### Public variant wrappers (`generate_variant_*`)

Six thin public functions — `generate_variant_a` through `generate_variant_f` — share a common pattern:

```python
def generate_variant_X(
    input_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> None
```

**What each does:** Loads the source NIfTI with `_load`, delegates to the corresponding `_variant_X_from_data` helper, and wraps the entire call in a broad `except Exception` that logs the error and returns silently.

**Design rationale:** The `try/except` at this level means a single corrupt or incompatible input file does not abort the entire dataset generation job. Errors are logged via the module-level `logger` and the caller (typically `dataset_gen.py`) continues to the next file. This is appropriate for batch processing but means failures are silent to any caller that does not inspect logs.

**Parameters:**
- `input_path` (`Path`) — source `.nii.gz` file.
- `output_path` (`Path`) — destination `.nii.gz` file.
- `target_shape` (`Tuple[int, int, int]`, default `TARGET_SHAPE`) — output voxel grid.

**Returns:** `None`. Side effect: writes one `.nii.gz` file on success; logs an error and writes nothing on failure.

---

### Resize / resampling strategy summary

| Operation | MONAI call | Library-agnostic? |
|---|---|---|
| Isotropic physical-space resample | `Spacing(pixdim, mode="bilinear")` via `MetaTensor` | No — tightly coupled to MONAI and PyTorch |
| Geometric resize to fixed grid | `Resize(target_shape, mode="trilinear")` | No — MONAI; could be replaced with `scipy.ndimage.zoom` |
| Centre crop / zero-pad to fixed grid | `ResizeWithPadOrCrop(shape)` | No — MONAI; crop is pure slicing, pad is zero-fill |

All interpolating operations use bilinear/trilinear mode. No anti-aliasing is applied before downsampling. Each variant is designed to require exactly **one interpolation pass** per volume.

---

### Control flow notes and flags

1. **Silent failure in public wrappers.** The broad `except Exception` in every `generate_variant_*` function swallows all errors after logging. There is no return value or status flag, so callers cannot programmatically distinguish success from failure without parsing logs.

2. **Redundant `crop_shape` computation in `_variant_c_from_data`.** The function recomputes `crop_shape = tuple(int(s * CROP_FRACTION) for s in data.shape)` for the affine scaling step, but `_center_crop` performs the identical computation internally. If `_center_crop` were modified to also return the crop shape it used, this duplication would disappear.

3. **`KeyError` on unregistered target shapes in E and F.** `_variant_e_from_data` and `_variant_f_from_data` index `E_SPACING_MM_BY_SHAPE` and `F_SPACING_MM_BY_SHAPE` with `target_shape`. If a new target shape is added to `VALID_TARGET_SHAPES` without a matching entry in these dicts, the result is a `KeyError` caught silently by the public wrapper — a confusing failure mode. An explicit validation guard at the top of these helpers (or at module import time) would make this contract visible.

4. **Affine column-norm assumption.** `_dynamic_spacing` extracts voxel sizes as `np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))`. This is correct for NIfTI affines that encode spacing as pure scaling (possibly with axis flips), but would give incorrect spacings for affines that include shear. Clinical NIfTI data is rarely sheared, so this is unlikely to be a practical issue, but it is undocumented.

5. **`MONAI` channel-first convention throughout.** Every transform call adds a leading channel dimension (`data[np.newaxis]`) and strips it on return (`result[0]`). This is consistent and correct, but results in repetitive boilerplate across all six `_variant_*_from_data` functions. A single helper that wraps any transform call with channel-add / channel-strip could reduce this noise.
