## data_cleaner.py

**Location:** `data_engine/data_cleaner.py`

The top-level entry point for the DICOM ingestion and NIfTI standardisation pipeline. It discovers raw DICOM case folders, validates them for geometric integrity, attaches clinical ground-truth labels, converts eligible series to NIfTI, and writes a full audit trail covering every input case.

---

### CLI interface

```
python data_cleaner.py [--raw-dir PATH] [--nifti-dir PATH] [--workers N] [--log-dir PATH]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--raw-dir` | path | `/mnt/data/cases-3/raw` | Root directory containing raw DICOM case folders (one subfolder per patient). |
| `--nifti-dir` | path | `/mnt/data/cases-3/nifti` | Output directory for converted `.nii.gz` files. Must be on a mounted filesystem (see mount guard below). |
| `--workers` | int | `os.cpu_count()` | Number of parallel subprocesses for NIfTI conversion. Reduce when RAM is constrained: each worker loads one full 3D volume. |
| `--log-dir` | path | `data_engine/output/` | Directory for all output CSVs and the JSONL log file. Override to isolate logs from a supplementary batch run without overwriting the previous run's outputs. |

---

### Disk state required before running

| Path | What must exist |
|---|---|
| `--raw-dir` | A directory of patient case folders named in the `bp{NNN}` / `BP{NNN}` convention, each containing `.dcm` files (directly or one subfolder deep). |
| `--nifti-dir` (parent) | Must resolve to a mount point that is not the filesystem root (`/`). The pipeline raises `RuntimeError` immediately if only `/` is found — this is a deliberate guard against silent fallback to the local disk when the external SSD is offline. |
| `data_engine/dataset/classes.csv` | Hard-coded path; not overridable via CLI. Must contain columns `exam`, `class`, `location`, `Age`. If absent the class-join step is skipped with a warning and all cases will later be flagged `MISSING_CLASS`. |
| `data_engine/output/` (or `--log-dir`) | Created automatically by `run_pipeline` before the logger is initialised. |

---

### Overall pipeline flow

`run_pipeline` executes nine sequential steps. All state is threaded through a shared list of case-record dicts; each step mutates those dicts in place or appends new keys.

```
get_subfolders()
    └─ organize_data()          # name normalisation + folder layout classification
        └─ validate_dcms()      # DICOM integrity and geometry checks
            └─ analyze_mixed_folders()   # side-output only; no state change
                └─ join_class_data()     # attach labels from classes.csv
                    └─ check_missing_class()   # flag OK cases with no label
                        └─ filter_for_conversion()   # keep only OK + labelled
                            └─ process_and_convert_exams()   # parallel NIfTI write
                                └─ _build_audit_log()        # merge pre/post-conversion records
                                    └─ write_audit_log()     # write ingestion_summary.csv
                                        └─ csv.DictWriter    # write folder_rename_map.csv
```

The entire body of `run_pipeline` is wrapped in a single `try/except Exception` block. On any unhandled exception the logger records the error with a full traceback and then re-raises, so the process exits non-zero and Slurm/shell callers can detect the failure.

---

### Module-level constants

These are resolved relative to `data_cleaner.py` at import time and remain fixed for the lifetime of the process (they are not CLI-overridable):

| Constant | Value | Purpose |
|---|---|---|
| `BASE_DIR` | `data_engine/` | Anchor for all relative output paths. |
| `DEFAULT_RAW_DATA_PATH` | `/mnt/data/cases-3/raw` | Default `--raw-dir`. |
| `DEFAULT_OUTPUT_NIFTI_PATH` | `/mnt/data/cases-3/nifti` | Default `--nifti-dir`. |
| `OUTPUT_DIR` | `data_engine/output/` | Default log/CSV output directory. |
| `DATASET_DIR` | `data_engine/dataset/` | Location of `classes.csv`. |
| `CLASSES_CSV_PATH` | `data_engine/dataset/classes.csv` | Hard-coded; not parameterised. |

Note: `VALIDATION_SUMMARY_CSV_PATH`, `MIXED_SERIES_CSV_PATH`, `INGESTION_LOG_PATH`, and `AUDIT_LOG_PATH` are defined at module level but are superseded inside `run_pipeline` when `--log-dir` is provided. The module-level definitions are dead code for any caller that passes `log_dir`.

---

### Functions in `data_cleaner.py`

#### `_build_audit_log(conversion_results, final_data) -> list[dict]`

Constructs a single audit list that accounts for every input case, not just those that reached NIfTI conversion.

- **Why it exists:** SC-005 traceability requirement — every input series must have a recorded outcome. Cases rejected before conversion (failed validation, missing label, etc.) never appear in `conversion_results`, so they must be back-filled.
- **Parameters:**
  - `conversion_results` — list of result dicts returned by `process_and_convert_exams`; each has keys `exam_name`, `status`, `reason`, `output_path`.
  - `final_data` — the full case-record list after all pre-conversion steps, including cases that were filtered out.
- **Returns:** A new list combining `conversion_results` with synthesised `failed` entries for every case whose `fixed_name` does not appear in `conversion_results`. The synthesised entries use the case's `validation_status` as the failure reason.
- **Side effects:** None. Reads but does not modify either input list.

#### `run_pipeline(raw_dir, nifti_dir, max_workers=None, log_dir=None) -> None`

Executes the full pipeline from DICOM discovery to audit log write.

- **Parameters:**
  - `raw_dir` — `str | Path` to the raw DICOM root.
  - `nifti_dir` — `str | Path` to the NIfTI output root.
  - `max_workers` — passed directly to `ProcessPoolExecutor`; `None` uses `os.cpu_count()`.
  - `log_dir` — if given, all four output files are written there instead of `OUTPUT_DIR`. The directory is created with `mkdir(parents=True, exist_ok=True)` before the logger opens its file handle.
- **Returns:** `None`.
- **Side effects (filesystem writes):**
  - `{log_dir}/ingestion.log` — JSONL runtime log, written incrementally during the run.
  - `{log_dir}/mixed_series_analysis.csv` — per-series breakdown for folders with multiple SeriesInstanceUIDs; written only if mixed cases exist.
  - `{log_dir}/ingestion_summary.csv` — conversion outcome for every input case (`write_audit_log`).
  - `{log_dir}/folder_rename_map.csv` — per-case validation summary including name mapping, DICOM metadata, and classification fields.
  - `{nifti_dir}/{class}/{fixed_name}.nii.gz` — one file per successfully converted case, placed in `0/` or `1/` subdirectories.
- **Mount guard:** walks `nifti_dir` upward to find the deepest real mount point. If only `/` is found it raises `RuntimeError` before any work begins. Prevents silent output to the local disk when the external SSD is unplugged.

---

### `src/file_utils.py`

Handles DICOM folder discovery and name standardisation (pipeline stage 1).

#### `get_subfolders(path) -> list[str]`

Returns names of all immediate subdirectories inside `path` using a single `os.scandir` call (avoids a second `stat()` per entry). Returns `[]` and logs an error on `FileNotFoundError` rather than raising, so a missing root directory does not abort the pipeline.

- **Parameters:** `path` — absolute path to scan.
- **Returns:** List of subdirectory names (not full paths). Order is filesystem-dependent.
- **Side effects:** Logs an error if the path does not exist.

#### `count_dcm_files(path) -> int`

Counts `.dcm` files (case-insensitive) at the top level of `path` only — deliberately non-recursive so the caller can distinguish between files directly in a folder and files in subfolders. Returns `0` on any filesystem error.

- **Parameters:** `path` — directory to count in.
- **Returns:** Integer count.
- **Side effects:** Logs an error on `FileNotFoundError` or `OSError`.

#### `generate_new_names(folder_list) -> list[dict]`

Maps raw folder names to canonical `BP{NNN}` names (zero-padded to 3 digits). Handles the `BP_001` variant (optional underscore). Non-matching names pass through unchanged. Sorts input before processing so duplicate suffixes (`A`, `B`, `C`) are assigned deterministically regardless of filesystem ordering, making the rename map stable across runs.

- **Parameters:** `folder_list` — raw folder names as strings.
- **Returns:** Sorted list of `{'original_name': ..., 'fixed_name': ...}` dicts.
- **Side effects:** None.
- **Complex control flow:** The duplicate-suffix logic uses two `defaultdict` counters in sequence (a count pass then a suffix-assignment pass). This is correct but could be simplified to a single pass using `enumerate` over groups produced by `itertools.groupby`.

#### `find_missing_cases(name_mapping) -> list[dict]`

Scans the range BP001–BP999 and returns a record for each number absent from `name_mapping`. These records are appended to the main list so the audit log explicitly accounts for gaps in the dataset.

- **Parameters:** `name_mapping` — list of case records with `fixed_name` keys.
- **Returns:** List of `{'original_name': 'missing', 'fixed_name': 'BP{NNN}'}` dicts.
- **Side effects:** Logs an info message.
- **Note:** The hard-coded upper bound of 999 is implicit in `range(1, 1000)`. If the dataset ever exceeds 999 cases this will silently stop detecting gaps above that limit.

#### `get_folder_stats(base_path, folder_list) -> list[dict]`

Inspects each case folder to a maximum depth of two levels, counting direct `.dcm` files, non-empty subfolders, and `.dcm` files one level down. The shallow scan is intentional: standard PACS exports are never nested deeper, and a deep recursive scan would be slow.

- **Parameters:** `base_path` — absolute root path; `folder_list` — folder names relative to it.
- **Returns:** List of `{'folder', 'direct_items', 'non_empty_subfolders', 'items_in_subfolders'}` dicts.
- **Side effects:** Logs scan errors at INFO level (inconsistently — other error paths use ERROR).

#### `add_data_codes(name_mapping) -> list[dict]`

Assigns a `data_code` string to each record based on counts from `get_folder_stats`. The five codes are: `READY`, `SUBFOLDER_PATH`, `DUPLICATE_DATA`, `EMPTY`, `MISSING`.

- **Parameters:** `name_mapping` — records with optional `direct_items` / `non_empty_subfolders` keys.
- **Returns:** Same list with `data_code` added in place.
- **Side effects:** Logs progress messages.
- **Complex control flow:** A two-level `if/elif/else` tree on `non_empty_subfolders` and `direct_items`. The logic is correct and not excessively deep (three branches, two levels), but it could be expressed as a lookup table keyed on `(non_empty_subfolder_count > 1, non_empty_subfolder_count == 1, direct_items > 0)` if more codes are added later.

#### `add_data_paths(name_mapping) -> list[dict]`

Resolves the relative filesystem path that downstream steps should read. For `READY` cases the path is the original folder name; for `SUBFOLDER_PATH` it is `original_name/subfolder`; for all other codes the code string itself is used as a sentinel value (e.g. `'MISSING'`). Deletes `non_empty_subfolders` from each record after use to reduce memory and CSV bloat.

- **Parameters:** `name_mapping` — records with `data_code` and `non_empty_subfolders`.
- **Returns:** Same list with `data_path` added and `non_empty_subfolders` removed.
- **Side effects:** Mutates records in place; logs progress.

#### `organize_data(case_folders, RAW_DATA_PATH) -> list[dict]`

Orchestrates all `file_utils` sub-steps in order. The only function called directly by `data_cleaner.py`. Merges folder stats into the name mapping via a `dict`-of-stats indexed by folder name (O(1) lookup), then appends missing-case records, assigns codes, and resolves paths.

- **Parameters:** `case_folders` — raw folder names from `get_subfolders`; `RAW_DATA_PATH` — absolute root.
- **Returns:** Fully populated list of case-record dicts ready for DICOM validation.
- **Side effects:** Logs progress at each sub-step.

---

### `src/dicom_utils.py`

Validates each DICOM series for integrity and geometric correctness (pipeline stage 2), and produces an optional per-series breakdown of mixed-series folders (stage 3).

#### `get_orientation(iop) -> str`

Classifies a scan's imaging plane from the `ImageOrientationPatient` DICOM tag by rounding each of its six values to the nearest integer and comparing against known signatures for axial, coronal, and sagittal orientations. Rounding absorbs real-world scanner tilt without misclassifying genuinely oblique scans.

- **Parameters:** `iop` — list of 6 floats, or `None`.
- **Returns:** One of `'AXIAL'`, `'CORONAL'`, `'SAGITTAL'`, `'OBLIQUE'`, `'UNKNOWN'`.
- **Side effects:** None.
- **Complex control flow:** A four-branch `if/elif/else` on a rounded list. Adding more orientations would require extending this chain; a dict lookup on the tuple `tuple(rounded_iop)` would be more maintainable.

#### `load_dicom_metadata(path) -> list`

Reads DICOM headers (not pixel data) for all `.dcm` files in `path` using `pydicom.dcmread(..., stop_before_pixels=True)`. Skips corrupted files with a warning. Returns an empty list if all files fail.

- **Parameters:** `path` — absolute directory path.
- **Returns:** List of `pydicom.Dataset` objects.
- **Side effects:** Logs a warning per corrupted file.

#### `_sort_slices_by_projection(metadata_list, iop) -> None`

Sorts a DICOM series into anatomically correct order by projecting each slice's `ImagePositionPatient` coordinate onto the scan's stacking axis (computed as the cross product of the IOP row and column vectors). Adds a `.dist` float attribute (mm) to each Dataset. Modifies `metadata_list` in place.

- **Parameters:** `metadata_list` — list of Datasets; `iop` — IOP tag as 6 floats.
- **Returns:** `None` (in-place sort + attribute addition).
- **Side effects:** Mutates all objects in `metadata_list`.
- **Why not InstanceNumber:** PACS systems frequently reset or reorder `InstanceNumber` during export; physical coordinates are scanner-independent and reliable.

#### `_evaluate_spacing(metadata_list) -> tuple`

Analyses inter-slice distances (from `.dist` attributes set by `_sort_slices_by_projection`) to classify geometric integrity. Computes absolute differences between consecutive distances, rounds to 2 decimal places, and classifies:

- `DUPLICATE_SLICES` — any delta < 0.001 mm.
- `OK` — exactly one unique spacing and consecutive `InstanceNumber`s.
- `GAPPED_SEQUENCE` — exactly one unique spacing but non-consecutive `InstanceNumber`s.
- `GAPPED_SEQUENCE (N missing)` — two spacings where the larger is approximately 2× the smaller (within 0.1 mm); `N` is the count of double-spaced gaps.
- `VARIABLE_SPACING` — any other case with more than one unique spacing.

- **Parameters:** `metadata_list` — sorted list of Datasets with `.dist`.
- **Returns:** `(validation_status, unique_spacings, duplicate_count, deltas)`.
- **Side effects:** None.
- **Complex control flow:** A three-branch `if/elif/else` on `len(unique_spacings)`, with a nested branch inside the `len == 2` case. The `GAPPED_SEQUENCE` vs `OK` distinction inside the `len == 1` branch also reads `InstanceNumber`, making it the only spacing path that touches a non-geometric tag. This mixed concern is a candidate for separation.

#### `validate_dcms(data_mapping, base_path) -> list[dict]`

Main validation loop. Iterates every case record and runs up to nine ordered checks, stopping at the first failure. After all cases are processed, delegates to `_flag_exam_size_outliers` for a statistical pass.

Checks in order:
1. Skip non-processable codes (`EMPTY`, `MISSING`, `DUPLICATE_DATA`) → `NOT_APPLICABLE`.
2. No `.dcm` files at resolved path → `NO_DCM_FILES`.
3. All files corrupt → `ALL_DCMS_CORRUPT`.
4. Multiple `SeriesInstanceUID`s → `MIXED_SERIES_ERROR ({N} UIDs)`.
5. No `ORIGINAL` images after scout filter → `NO_ORIGINAL_IMAGES_FOUND`.
6. Only one slice → `NOT_3D_VOLUME_ERROR`.
7. Missing `IOP`, `IPP`, or `PixelSpacing` → `MISSING_SPATIAL_METADATA_ERROR`.
8. Projection-based sort.
9. Spacing evaluation → `OK`, `GAPPED_SEQUENCE`, `DUPLICATE_SLICES`, `VARIABLE_SPACING`.

After spacing evaluation: extracts `orientation`, `modality`, `patient_sex`, `image_dimensions`, `slice_thickness` (from the most common inter-slice delta, not the DICOM `SliceThickness` tag), and `exam_size` (slice count × effective thickness, in mm).

`PatientAge` is **not** extracted here; it comes from `classes.csv` via `class_utils.join_class_data`.

- **Parameters:** `data_mapping` — case records from `organize_data`; `base_path` — DICOM root.
- **Returns:** Same list with validation results merged in place.
- **Side effects:** Logs per-case progress and warnings for scout-filtered slices.
- **Complex control flow:** The nine-check sequence is implemented as a flat loop with early `continue`s — this is clear and idiomatic. The metadata extraction block at check 10 has a conditional on `isinstance(effective_slice_thickness, float)` that could produce `'N/A'` strings mixed with floats in `slice_thickness` and `exam_size`, complicating downstream comparisons.

#### `_flag_exam_size_outliers(data_mapping) -> list[dict]`

Applies fixed anatomical bounds (90 mm minimum, 300 mm maximum physical coverage) to all `OK` cases, re-classifying outliers as `BELOW_LIMIT` or `ABOVE_LIMIT`. Only considers cases already at `OK`; does not touch cases with other statuses.

- **Parameters:** `data_mapping` — full case list after `validate_dcms`.
- **Returns:** Same list with statuses updated in place.
- **Side effects:** Logs bound values and lists of flagged cases.
- **Note:** The docstring records that the original lower bound was IQR-based; it was replaced with a fixed 90 mm floor to avoid flagging valid cases when the dataset distribution changes. The constant is embedded in the function body with no named configuration point.

#### `analyze_mixed_series(folder_path) -> list[dict]`

Runs a mini-validation on each `SeriesInstanceUID` group within a mixed-series folder. Applies the same scout filter, orientation extraction, projection sort, and spacing analysis as the main pipeline. Produces a per-series report dict for each group.

- **Parameters:** `folder_path` — absolute path to the mixed case folder.
- **Returns:** List of per-series report dicts; `[]` if the folder is absent or unreadable.
- **Side effects:** None (read-only).
- **Overlap with `validate_dcms`:** The scout-filtering logic (`'ORIGINAL' in ImageType`), `get_orientation`, `_sort_slices_by_projection`, and `_evaluate_spacing` calls are duplicated verbatim from `validate_dcms`. The shared logic is not extracted into a helper, so any change to the main validation rules must be replicated here manually.

#### `analyze_mixed_folders(data_mapping, base_path, output_csv_path) -> None`

Collects all `MIXED_SERIES_ERROR` cases, calls `analyze_mixed_series` on each, and writes the combined report to `output_csv_path`. If no mixed cases exist the function returns immediately without creating the file. Column order is sorted alphabetically with `parent_case_name` pinned first.

- **Parameters:** `data_mapping` — full case list; `base_path` — DICOM root; `output_csv_path` — destination CSV path.
- **Returns:** `None`.
- **Side effects:** Creates `mixed_series_analysis.csv` if mixed cases exist. Logs progress.

---

### `src/class_utils.py`

Attaches ground-truth labels from `classes.csv` to case records (pipeline stage 4) and flags cases that passed validation but have no label (stage 5).

#### `join_class_data(validated_data, classes_csv_path) -> list[dict]`

Reads `classes.csv` into a dict keyed by uppercased exam name, then for each case record strips any alphabetic duplicate-disambiguation suffix (`BP001A` → `BP001`) before looking up the label. Adds `class`, `location`, and `patient_age` keys to matching records. If `classes.csv` is absent, logs a warning and returns the data unchanged.

- **Parameters:** `validated_data` — case records; `classes_csv_path` — path to the CSV.
- **Returns:** Same list with classification fields merged in place.
- **Side effects:** Logs progress and a warning on missing file.
- **Note:** `patient_age` is sourced here from the `Age` CSV column rather than from the DICOM `PatientAge` tag. This is a deliberate choice (noted in `validate_dcms`) because clinical records may be more accurate than PACS-exported headers, but it creates a dependency on `classes.csv` for a field that is available in the DICOM files themselves.

#### `check_missing_class(data) -> list[dict]`

Finds any case with `validation_status == 'OK'` and no `class` key, and updates its status to `MISSING_CLASS`. Cases with a non-OK status are untouched to preserve their original diagnostic reason in the audit log.

- **Parameters:** `data` — case records after `join_class_data`.
- **Returns:** Same list with statuses updated in place.
- **Side effects:** Logs the count of flagged cases.

---

### `src/nifti_utils.py`

Converts validated, labelled DICOM series to NIfTI volumes using MONAI + ITK (pipeline stages 6–7).

#### `filter_for_conversion(exam_data) -> list[dict]`

Selects cases where `validation_status == 'OK'` and `class in ('0', '1')`. Both conditions must hold; cases failing either are excluded silently (they are already recorded in the audit log under a non-OK status or `MISSING_CLASS`).

- **Parameters:** `exam_data` — full case list.
- **Returns:** Filtered list; does not modify the input.
- **Side effects:** Logs the eligible count.

#### `convert_series_to_nifti(dicom_dir, output_path) -> bool`

Loads a DICOM series directory using MONAI's `LoadImage(reader=ITKReader())`, casts pixel data to `float32`, constructs a `nibabel.Nifti1Image` with the ITK-derived affine matrix, and saves as `.nii.gz`. Explicitly deletes large objects and calls `gc.collect()` after saving to prevent memory accumulation in long-running subprocesses.

- **Parameters:** `dicom_dir` — absolute path to DICOM directory; `output_path` — destination `.nii.gz` path.
- **Returns:** `True` on success, `False` on non-IO exception.
- **Raises:** `OSError` on disk/permission failures (re-raised to the caller for distinct error handling).
- **Side effects:** Writes one `.nii.gz` file. Logs success or error.
- **Why ITKReader:** ITK handles DICOM series stacking, affine reconstruction, and multi-frame DICOM edge cases automatically. Implementing this correctly with raw pydicom would be fragile.

#### `_convert_exam_worker(exam, dicom_base_path, nifti_output_dir) -> dict`

Unit of work for each parallel subprocess. Constructs the output path as `{nifti_output_dir}/{class}/{fixed_name}.nii.gz`, creates the class subdirectory if needed, checks for existing output (skips if present for idempotency), and calls `convert_series_to_nifti`. Because it runs in a subprocess it cannot write to the main process logger directly; log messages are accumulated in `result['log_messages']` and flushed by the parent after the future completes.

- **Parameters:** `exam` — case record dict; `dicom_base_path`, `nifti_output_dir` — absolute root paths.
- **Returns:** Result dict with `exam_name`, `status` (`'success'` / `'skipped'` / `'failed'`), `reason`, `output_path`, `log_messages`.
- **Side effects:** Creates class subdirectory; writes `.nii.gz` file if not skipped.

#### `process_and_convert_exams(eligible_exams, dicom_base_path, nifti_output_dir, max_workers=None) -> list[dict]`

Submits all eligible exams to a `ProcessPoolExecutor` and collects results with `as_completed` for real-time progress logging. Flushes each worker's `log_messages` into the main logger after each future resolves, then removes that key before appending the result to the output list. Creates `nifti_output_dir` if it does not exist.

- **Parameters:** `eligible_exams` — filtered case list; `dicom_base_path`, `nifti_output_dir` — absolute paths; `max_workers` — passed to `ProcessPoolExecutor`.
- **Returns:** List of result dicts (without `log_messages`).
- **Side effects:** Creates `nifti_output_dir`; writes `.nii.gz` files via workers. Logs final success count.
- **Why subprocesses:** MONAI and ITK are C++ libraries with their own internal thread pools. Running them from Python threads causes contention; separate processes are fully isolated.

---

### `src/logging_utils.py`

Configures the shared `dicom_ingestion` named logger and writes the conversion audit CSV.

#### `_JsonlHandler` (class, internal)

Extends `logging.FileHandler`, overriding `emit()` to write each log record as a single-line JSON object (`timestamp`, `level`, `logger`, `message`, optional `exception`). Calls `self.flush()` after every write so records survive a mid-run crash. Not part of the public API.

#### `setup_logger(log_path) -> logging.Logger`

Initialises the `dicom_ingestion` named logger with two handlers: a `_JsonlHandler` at DEBUG level writing to `log_path`, and a `StreamHandler` at INFO level writing to the console. Guards against double-initialisation with `if logger.handlers: return logger` (important for test isolation). All `src/` modules obtain the same logger instance via `logging.getLogger("dicom_ingestion")`.

- **Parameters:** `log_path` — path for the JSONL file.
- **Returns:** Configured logger.
- **Side effects:** Creates the JSONL log file on disk.

#### `write_audit_log(results, output_path) -> None`

Writes the conversion outcome list to a four-column CSV (`exam_name`, `status`, `reason`, `output_path`). Overwrites the file if it already exists. Logs but does not raise on write failure.

- **Parameters:** `results` — list of result dicts from `_build_audit_log`; `output_path` — destination path.
- **Returns:** `None`.
- **Side effects:** Writes `ingestion_summary.csv`.

#### `log_jsonl(data, log_path) -> None`

Standalone utility (not called by the main pipeline) that appends a single dict as a JSON line to a JSONL file, adding a `timestamp` key before writing. Available for future ad-hoc metrics logging. Modifies the `data` dict in place by adding `timestamp`.

- **Parameters:** `data` — dict to serialise; `log_path` — JSONL file path (opened in append mode).
- **Returns:** `None`.
- **Side effects:** Appends one line to `log_path`; mutates `data`.

---

### Output files produced per run

| File | Written by | Contents |
|---|---|---|
| `{log_dir}/ingestion.log` | `_JsonlHandler` (incremental) | JSONL runtime log, all DEBUG+ events. |
| `{log_dir}/mixed_series_analysis.csv` | `analyze_mixed_folders` | Per-series breakdown for MIXED_SERIES_ERROR cases; created only if such cases exist. |
| `{log_dir}/ingestion_summary.csv` | `write_audit_log` | One row per input case: final outcome (success / skipped / failed) and reason. |
| `{log_dir}/folder_rename_map.csv` | `data_cleaner.run_pipeline` (inline) | One row per case: name mapping, DICOM metadata, classification fields, validation status. |
| `{nifti_dir}/0/*.nii.gz` | `_convert_exam_worker` | Converted healthy cases. |
| `{nifti_dir}/1/*.nii.gz` | `_convert_exam_worker` | Converted aneurysm cases. |

---

### Flags: complex control flow

1. **`_evaluate_spacing` — mixed geometric and instance-number logic.** The `len(unique_spacings) == 1` branch additionally reads `InstanceNumber` to distinguish `OK` from `GAPPED_SEQUENCE`. This mixes two concerns (physical spacing and sequence numbering) in a single function. Consider separating into `_classify_physical_spacing` and a wrapper that applies the instance-number refinement.

2. **`validate_dcms` — nine-check early-exit loop.** The sequence of `continue`s is readable but the function is long (~110 lines of body). The metadata extraction block at the end (step 10) uses `isinstance(effective_slice_thickness, float)` to gate two different code paths, which means `slice_thickness` and `exam_size` can be either a numeric type or the string `'N/A'`. Downstream consumers (`_flag_exam_size_outliers`) must defensively catch `ValueError`/`TypeError` when converting these fields.

3. **`add_data_codes` — two-pass duplicate suffix assignment.** `generate_new_names` makes two passes over the name list (one to count, one to assign suffixes). A single pass grouping by canonical name would achieve the same result with less state.

4. **`get_orientation` — `if/elif` chain over a list value.** As currently written, adding a new orientation requires editing the chain. A dict lookup on `tuple(rounded_iop)` would be O(1) and trivially extensible.

---

### Flags: logic overlapping with other scripts

1. **`analyze_mixed_series` duplicates validation logic from `validate_dcms`.** The scout-filtering predicate (`'ORIGINAL' in ImageType`), calls to `get_orientation`, `_sort_slices_by_projection`, and `_evaluate_spacing` appear verbatim in both functions. If the validation rules change (e.g. a different scout-detection heuristic), both sites must be updated in sync. Extracting a `_validate_single_series(metadata_list) -> dict` helper and calling it from both would eliminate this duplication.

2. **`data_cleaner.py` writes `folder_rename_map.csv` inline.** The CSV write at pipeline step 9 (lines 160–169 of `data_cleaner.py`) is the only I/O operation not delegated to a `src/` module — all other file writes happen inside `logging_utils`, `dicom_utils`, or `nifti_utils`. Moving it to `logging_utils.write_validation_summary` would make `run_pipeline` a pure orchestrator with no direct I/O.

3. **`dataset_gen.py` reads the NIfTI output that `data_cleaner.py` produces.** There is no functional overlap between the two scripts, but both hard-code `/mnt/data/cases-3/` as their default base path. A shared config file or environment variable would prevent these from drifting apart.

4. **`CLASSES_CSV_PATH` is hard-coded and not CLI-overridable.** Every other input/output path is parameterisable via `--raw-dir`, `--nifti-dir`, and `--log-dir`, but the path to `classes.csv` is fixed at `data_engine/dataset/classes.csv`. If a separate batch of cases has its own label file, the user must either modify the source or symlink the file. Adding `--classes-csv` to the CLI would be consistent with the existing design.
