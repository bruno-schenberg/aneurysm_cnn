## dataset_gen.py

`data_engine/dataset_gen.py` is the CLI entry point for generating all preprocessing variant datasets from a directory of NIfTI volumes. It reads `.nii.gz` files produced by the upstream DICOM-to-NIfTI conversion step, applies six distinct spatial preprocessing strategies (variants A–F), and writes the results into separate output directories — one directory per variant. Processing is parallelised at the file level using `multiprocessing`, with one subprocess per file and a dedicated pool per variant.

---

### Inputs and outputs

**Input:** A directory of `.nii.gz` files organised into class subdirectories (`0/` and `1/`). The script first searches for files matching `*/*.nii.gz` (class subdirectory layout) and falls back to a flat `*.nii.gz` glob at the top level. The class subdirectory name is preserved in the output.

**Output:** For each requested variant and target shape, a subdirectory is created under the output base directory. The filename of each output file is identical to the input filename. The output directory tree mirrors the input class structure:

```
<output_base>/
  dataset_A192/        # variant A at 192x192x128
    0/BP001.nii.gz
    1/BP002.nii.gz
    ...
  dataset_B192/
    ...
```

**File format:** All output files are compressed NIfTI (`.nii.gz`), float32, with an updated affine matrix appropriate to each variant's transform.

---

### Module-level constants

| Name | Value | Notes |
|---|---|---|
| `DEFAULT_INPUT_DIR` | `/mnt/data/cases-3/nifti` | Hardcoded local path; always override with `--input-dir` on HPC. |
| `DEFAULT_OUTPUT_BASE` | `/mnt/data/cases-3` | Hardcoded local path; always override with `--output-base` on HPC. |
| `DEFAULT_WORKERS` | `4` | Workers per variant pool. |
| `OUTPUT_FOLDER_NAMES` | nested dict | Maps `(shape_key, variant_key)` to output subfolder name. See below. |

`OUTPUT_FOLDER_NAMES` encodes all output directory names as a two-level dict keyed first by resolution string, then by variant letter. There are currently three resolutions and six variants (18 entries). Adding a new resolution requires adding a matching entry here and in `VALID_TARGET_SHAPES` in `nifti_resize.py`.

The four thread-count environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) are set to `"1"` at the very top of the file, before any library import. This is intentional and load-order-critical: scipy, numpy, and MONAI read these variables when their C extensions are first loaded; setting them after import has no effect. This prevents the C-extension thread pools from competing with Python's `multiprocessing` workers.

---

### CLI interface

```
python -m data_engine.dataset_gen [OPTIONS]
# or
python data_engine/dataset_gen.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--input-dir DIR` | `/mnt/data/cases-3/nifti` | Root directory containing `.nii.gz` source files, optionally in `0/`/`1/` subdirs. |
| `--output-base DIR` | `/mnt/data/cases-3` | Base directory under which all variant dataset subfolders are created. |
| `--target-shape SHAPE` | `192x192x128` | Output voxel grid. Choices: `128x128x128`, `192x192x128`, `256x256x176`. |
| `--workers N` | `4` | Number of parallel worker processes per variant. |
| `--seed N` | `42` | RNG seed for shuffling the file list before processing. |
| `--variants V [V ...]` | `C D A B E F` | Subset and order of variants to generate. Choices: `A B C D E F`. C and D run first by default as they are cheaper and serve as a smoke test. |

The script exits with code 1 if the input directory does not exist, if no `.nii.gz` files are found, or if any file fails processing across any variant. On success it exits with code 0.

---

### Overall flow

1. Parse arguments and configure the root logger.
2. Validate that `--input-dir` exists.
3. Resolve the target shape tuple from `VALID_TARGET_SHAPES` using the shape key string.
4. Collect all `.nii.gz` files (subdirectory layout preferred, flat fallback).
5. Shuffle the file list with the given seed. This distributes files across workers in a randomised order, which is useful for progress estimation and avoids all large files landing in one worker.
6. Clamp `n_workers` to `min(args.workers, len(nifti_files))`.
7. For each requested variant key in order, call `_run_variant`. Variants are strictly sequential; the next variant's pool does not start until the current one finishes.
8. Accumulate all failures across all variants.
9. If any failures occurred, log them all and exit with code 1. Otherwise log completion and exit with code 0.

---

### Functions — `dataset_gen.py`

#### `_worker_init() -> None`

**What it does:** Initialiser function called once per worker process at pool startup. Sets PyTorch's intra-op thread count to 1 and configures the worker's `logging.basicConfig` with a format that includes the process name.

**Why written this way:** Worker processes are spawned fresh (via `fork`) and do not inherit the main process's logging configuration. Without this, log output from workers is lost. `torch.set_num_threads(1)` prevents PyTorch from spawning its own thread pools inside a worker that is itself one of many parallel processes.

**Parameters:** None. **Returns:** None. **Side effects:** Mutates the PyTorch thread count and installs a log handler in the worker process.

---

#### `_task_a` through `_task_f`

Six functions with identical structure, one per variant. Each is a top-level module function (not a closure or lambda).

**What they do:** Each accepts a single `args` tuple of `(file_path, output_path, target_shape)`, skips the file if the output already exists (idempotency), loads the NIfTI volume via `_load`, calls the corresponding `_variant_X_from_data` transform function, then explicitly deletes the loaded arrays and calls `gc.collect()`.

**Return value:** `tuple[str, str | None]` — a human-readable label (`"filename.nii.gz:A"`) and either `None` on success or an error string of the form `"ExceptionType: message"` on failure.

**Why written this way:** These functions must be defined at module level rather than as closures or `functools.partial` wrappers because the `fork` start method requires that worker callables be importable by name from a clean process state. The six separate functions also avoid any dictionary lookup or dispatch inside the hot path. The single-tuple argument signature is required by `pool.imap_unordered`, which passes one positional argument per task. Using a tuple instead of multiple arguments keeps the call compatible with `imap_unordered` without needing `starmap`.

**Side effects:** Writes one `.nii.gz` file to disk per successful call. Calls `gc.collect()` after processing to release memory held by C-extension heaps (MONAI/ITK).

**Idempotency:** If the output file already exists the function returns immediately with `(label, None)`, making reruns safe after interruption.

---

#### `_TASK_FN` (module-level dict)

Maps variant key strings `"A"` through `"F"` to the corresponding `_task_X` function. Used by `_run_variant` to select the correct worker function without a chain of `if`/`elif`.

---

#### `_run_variant(variant_key, nifti_files, n_workers, target_shape, target_shape_key, output_base) -> list[tuple[str, str]]`

**What it does:** Orchestrates parallel processing of all input files for one variant. Builds the task list, creates a fresh `fork` process pool, dispatches tasks with `imap_unordered`, logs per-file progress, and returns the list of `(label, error)` pairs for any failed files.

**Parameters:**
- `variant_key: str` — one of `"A"` through `"F"`.
- `nifti_files: list[Path]` — flat list of source `.nii.gz` paths.
- `n_workers: int` — pool size.
- `target_shape: tuple[int, int, int]` — voxel grid tuple passed into each task.
- `target_shape_key: str` — string key used to look up the output folder name (e.g. `"192x192x128"`).
- `output_base: Path` — base directory under which the variant's output folder is created.

**Returns:** `list[tuple[str, str]]` — one entry per failed file; empty on full success.

**Why written this way:**
- A fresh pool is created for each variant so that memory and state from one variant's workers cannot leak into the next.
- `maxtasksperchild=1` is set so that the OS reclaims each worker's entire address space (including any C-extension heap) immediately after the worker finishes its single file. This prevents per-process memory from accumulating across the full file list.
- `multiprocessing.get_context("fork")` is called explicitly. Without it, PyTorch or MONAI may have already called `set_start_method("forkserver")`, which spawns workers from a clean server process that does not inherit `sys.path`. Under `forkserver` the `nifti_resize` imports silently fail and tasks stall.
- `imap_unordered` is used instead of `map` so that results are logged as soon as each file finishes, giving real-time progress feedback.

**Side effects:** Creates output subdirectories (`out_base / class_dir`) via `mkdir(parents=True, exist_ok=True)`. Logs progress at `INFO` level and failures at `WARNING`/`ERROR`. Spawns and joins worker processes.

---

#### `main() -> None`

**What it does:** Entry point. Parses CLI arguments, validates inputs, collects files, shuffles them, then calls `_run_variant` once per requested variant in sequence. Aggregates all failures and exits with code 1 if any occurred.

**Parameters:** None (reads from `sys.argv`). **Returns:** None.

**Side effects:** Writes all output dataset directories and files to disk. Logs to stdout/stderr. Calls `sys.exit(1)` on validation errors or any processing failure.

---

### Functions — `src/nifti_resize.py`

This module contains all transform logic. It is imported by `dataset_gen.py` and has no CLI of its own.

#### Module-level constants

| Name | Value / type | Purpose |
|---|---|---|
| `TARGET_SHAPE` | `(128, 128, 128)` | Legacy default; kept for backward compatibility with public variant functions. Not used by `dataset_gen.py`. |
| `CROP_FRACTION` | `0.8` | Fraction of each axis retained by the centre-crop step (variants A, C, E). |
| `VALID_TARGET_SHAPES` | `dict[str, tuple]` | Maps resolution key strings to voxel grid tuples. Must be kept in sync with `OUTPUT_FOLDER_NAMES` in `dataset_gen.py`. |
| `E_SPACING_MM_BY_SHAPE` | `dict[tuple, float]` | Fixed isotropic spacing for variant E, derived as `192.0 / target_XY` (largest 80%-cropped FOV is 192 mm). |
| `F_SPACING_MM_BY_SHAPE` | `dict[tuple, float]` | Fixed isotropic spacing for variant F, derived as `240.0 / target_XY` (largest full FOV in the dataset is 240 mm). |

The 240 mm and 192 mm maximum FOV values used to derive E and F spacings are hardcoded from dataset inspection and are not configurable at runtime. If the dataset grows to include scans with larger FOVs these constants must be updated manually.

---

#### `_load(path: Path) -> tuple[np.ndarray, np.ndarray]`

**What it does:** Loads a `.nii.gz` file and returns `(data, affine)` where `data` is a float32 array and `affine` is a `(4, 4)` matrix.

**Why written this way:** `mmap=False` forces NiBabel to read the full file into RAM rather than memory-mapping it, avoiding file-handle contention across parallel worker processes. `get_fdata(dtype=np.float32)` casts on load, halving memory footprint versus the default float64.

**Parameters:** `path: Path`. **Returns:** `tuple[np.ndarray, np.ndarray]`. **Side effects:** Reads from disk.

---

#### `_save(data, affine, path, target_shape) -> None`

**What it does:** Asserts that `data.shape == target_shape`, then saves a `Nifti1Image` to the given path via `nib.save`.

**Why written this way:** The assertion is a deliberate early-fail guard. Without it, a bug in a variant function that produces the wrong shape would silently write a malformed file that would only fail much later during training. The assertion surfaces the error immediately with a clear message.

**Parameters:** `data: np.ndarray`, `affine: np.ndarray`, `path: Path`, `target_shape: tuple`. **Returns:** None. **Side effects:** Writes a `.nii.gz` file to disk.

---

#### `_center_crop(data, crop_fraction) -> np.ndarray`

**What it does:** Returns the central `crop_fraction` of each axis. Crops to `(int(H*f), int(W*f), int(D*f))`. Uses MONAI's `ResizeWithPadOrCrop`, which performs pure array slicing (no interpolation) when the target is smaller than the input.

**Parameters:** `data: np.ndarray`, `crop_fraction: float` (default `CROP_FRACTION = 0.8`). **Returns:** Cropped `np.ndarray`. **Side effects:** None.

---

#### `_dynamic_spacing(data, affine, target_shape) -> float`

**What it does:** Computes the isotropic voxel spacing (in mm) that maps the largest physical axis of the volume to the corresponding target dimension without cropping.

**Formula:** `spacing = max_i(physical_extent_i / target_i)` where `physical_extent_i = n_voxels_i * zoom_i`. Zooms are extracted from the affine via `np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))`.

**Why written this way:** By taking the maximum ratio across axes, this guarantees the resampled volume fits within `target_shape` on every axis simultaneously. Axes that land below the target after resampling are zero-padded by `ResizeWithPadOrCrop`.

**Parameters:** `data: np.ndarray`, `affine: np.ndarray`, `target_shape: tuple`. **Returns:** `float`. **Side effects:** None.

---

#### `_resample_isotropic(data, affine, spacing) -> tuple[np.ndarray, np.ndarray]`

**What it does:** Resamples a volume to a given isotropic voxel spacing using MONAI's `Spacing` transform with bilinear interpolation. Returns `(resampled_data, new_affine)`.

**Why written this way:** MONAI's `Spacing` transform operates on `MetaTensor` objects that carry the affine. Wrapping the numpy array in a `MetaTensor` is the idiomatic MONAI pattern; it avoids manual affine recalculation and ensures the transform updates the affine correctly. The affine is passed as float64 because MONAI's `Spacing` internally uses float64 for spatial calculations.

**Parameters:** `data: np.ndarray`, `affine: np.ndarray`, `spacing: float`. **Returns:** `tuple[np.ndarray, np.ndarray]`. **Side effects:** None (does not read or write files). Allocates a new volume and affine; deletes intermediate `MetaTensor` and result objects explicitly.

---

#### `_variant_a_from_data(data, affine, output_path, target_shape) -> None`

**Pipeline:** `_center_crop` (80%) → `_dynamic_spacing` on cropped volume → `_resample_isotropic` → `ResizeWithPadOrCrop` to `target_shape` → `_save`.

The dynamic spacing is computed on the cropped volume, not the original, so the cropped anatomy is guaranteed to fit within `target_shape` without a second crop step.

**Parameters:** pre-loaded `data`, `affine`, destination `output_path`, `target_shape`. **Returns:** None. **Side effects:** Writes one `.nii.gz` file.

---

#### `_variant_b_from_data(data, affine, output_path, target_shape) -> None`

**Pipeline:** `_dynamic_spacing` on full volume → `_resample_isotropic` → `ResizeWithPadOrCrop` → `_save`.

Like A but operates on the full native volume rather than the cropped sub-volume.

**Parameters / returns / side effects:** same pattern as A.

---

#### `_variant_c_from_data(data, affine, output_path, target_shape) -> None`

**Pipeline:** `_center_crop` (80%) → MONAI `Resize` (trilinear) to `target_shape` → affine column scaling → `_save`.

No resampling step. The affine is updated by scaling the rotation/scale columns by `crop_shape / target_shape` to reflect the new effective voxel size after the trilinear resize.

**Parameters / returns / side effects:** same pattern as A.

---

#### `_variant_d_from_data(data, affine, output_path, target_shape) -> None`

**Pipeline:** MONAI `Resize` (trilinear) on full volume to `target_shape` → affine column scaling → `_save`.

The simplest variant: a single trilinear resize of the full native volume. Affine is scaled by `original_shape / target_shape`.

**Parameters / returns / side effects:** same pattern as A.

---

#### `_variant_e_from_data(data, affine, output_path, target_shape) -> None`

**Pipeline:** `_center_crop` (80%) → `_resample_isotropic` at fixed spacing from `E_SPACING_MM_BY_SHAPE[target_shape]` → `ResizeWithPadOrCrop` → `_save`.

Like A but uses a fixed per-shape spacing rather than a per-scan dynamic spacing. The fixed spacing is derived from the largest 80%-cropped FOV in the dataset (192 mm), guaranteeing no second crop for any scan.

**Parameters / returns / side effects:** same pattern as A.

---

#### `_variant_f_from_data(data, affine, output_path, target_shape) -> None`

**Pipeline:** `_resample_isotropic` at fixed spacing from `F_SPACING_MM_BY_SHAPE[target_shape]` → `ResizeWithPadOrCrop` → `_save`.

Like B but with a fixed scanner-independent spacing. Every output voxel represents the same physical volume of tissue regardless of the source scanner.

**Parameters / returns / side effects:** same pattern as A.

---

#### Public `generate_variant_X` functions (six functions)

Thin wrappers around the corresponding `_variant_X_from_data` functions. Each accepts `(input_path, output_path, target_shape)`, calls `_load`, and delegates to the private form. Errors are caught and logged rather than re-raised. These are the public API of `nifti_resize.py`; `dataset_gen.py` does not use them (it calls the `_from_data` forms directly after loading once per file).

---

### Variant design summary

The six variants form a 3x2 factorial design:

|  | 80% centre crop | Full volume |
|---|---|---|
| No resampling | C | D |
| Dynamic isotropic resample | A | B |
| Fixed isotropic resample | E | F |

This structure enables two experimental comparisons:
- **Crop vs. no-crop (rows):** A vs. B, C vs. D, E vs. F — does discarding the outer 20% of anatomy improve classification?
- **Resampling strategy (columns):** C/D vs. A/B vs. E/F — does normalising to isotropic spacing help generalisation, and does a fixed (scanner-independent) spacing outperform a per-scan dynamic one?

---

### Hardcoded values that should be configurable

| Location | Value | Issue |
|---|---|---|
| `dataset_gen.py:35` | `DEFAULT_INPUT_DIR = Path("/mnt/data/cases-3/nifti")` | Machine-specific path. Usable only with `--input-dir` override on any non-local machine. |
| `dataset_gen.py:36` | `DEFAULT_OUTPUT_BASE = Path("/mnt/data/cases-3")` | Same issue. |
| `nifti_resize.py:97–109` | `192.0` and `240.0` in `E_SPACING_MM_BY_SHAPE` / `F_SPACING_MM_BY_SHAPE` | These encode the maximum 80%-cropped and full FOV observed in the dataset. If the dataset grows to include scans with larger FOVs, the spacings become too coarse and some scans will be cropped during padding. There is no runtime check or warning for this. |
| `nifti_resize.py:82` | `CROP_FRACTION = 0.8` | Not exposed as a CLI argument. Changing it requires editing source. |

---

### Complex control flow and simplification opportunities

**Six nearly identical `_task_X` functions (lines 101–198):** The six worker functions are structurally identical — only the label suffix and the `_variant_X_from_data` call differ. They were intentionally kept as six separate module-level functions to satisfy picklability requirements under the `fork` start method. A factory function or decorator that generates them at module load time would preserve picklability while eliminating the repetition. The `_TASK_FN` dict already abstracts selection; the body duplication is the remaining concern.

**File discovery fallback (lines 364–369):** A two-step glob with a silent warning fallback (`rglob("*/*.nii.gz")` then `glob("*.nii.gz")`) could silently mix class-labelled and unlabelled files if the input directory has a non-standard layout. The fallback logs a warning but does not verify that the flat files are usable without class subdirectories (the class directory is inferred from `file_path.parent.name` in `_run_variant`, which would be the input directory name rather than `0` or `1` in this case).

**Sequential variant loop (lines 386–390):** Variants are run strictly sequentially. Since each file is loaded independently per variant (no shared state between variants), all six variants for a given file could in principle be computed in one pass (load once, apply all six transforms). The current design favours simplicity and pool isolation over throughput. At large scale this trades I/O (re-reading each file six times) for clean per-variant memory reclamation.
