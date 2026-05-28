## inventory_scan.py

### Purpose

`inventory_scan.py` is a read-only audit tool for a new batch of raw DICOM data. It extracts zip archives, runs the same organisation, validation, and label-join steps that `data_cleaner.py` runs, but deliberately stops before NIfTI conversion. The result is a single `inventory_summary.csv` that shows, for every expected case number in the BP001–BP999 range, whether the data is present, whether it is geometrically sound, whether it has a clinical label, and whether the PatientSex DICOM tag was found. As a side effect it writes any newly extracted PatientSex values back into `classes.csv`.

---

### Hardcoded constants and paths

| Name | Value | Role |
|---|---|---|
| `BASE_DIR` | `inventory_scan.py`'s own parent directory | Anchor for all relative paths |
| `DATASET_DIR` | `BASE_DIR / "dataset"` | Location of `classes.csv` |
| `OUTPUT_DIR` | `BASE_DIR / "output" / "inventory"` | Default output directory |
| `DEFAULT_ZIP_DIR` | `/mnt/data/raw` | Default source of `.zip` archives |
| `DEFAULT_EXTRACT_DIR` | `/mnt/data/raw_extract` | Default extraction target |
| `CLASSES_CSV_PATH` | `DATASET_DIR / "classes.csv"` | Label and gender source/sink |
| `SEX_TO_GENDER` | `{"M": 0, "F": 1}` | Numeric encoding written to `classes.csv` |
| `SUMMARY_FIELDNAMES` | 17-element list | Column order for the output CSV |
| `MIN_EXAM_SIZE_MM` (in `dicom_utils`) | `90.0` | Outlier floor for `BELOW_LIMIT` |
| `MAX_EXAM_SIZE_MM` (in `dicom_utils`) | `300.0` | Outlier ceiling for `ABOVE_LIMIT` |

All mount-point paths (`/mnt/data/raw`, `/mnt/data/raw_extract`) are hardcoded as defaults and must be overridden with CLI flags when the storage layout differs.

---

### CLI interface

```
python inventory_scan.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--zip-dir` | path | `/mnt/data/raw` | Directory containing `.zip` archives |
| `--extract-dir` | path | `/mnt/data/raw_extract` | Directory to extract archives into |
| `--partitioned` | flag | off | Process one zip at a time (low-disk mode) |
| `--skip-extract` | flag | off | Skip extraction, scan `--extract-dir` as-is (full mode only) |
| `--log-dir` | path | `data_engine/output/inventory/` | Output directory for CSVs and the log file |

`--skip-extract` and `--partitioned` are mutually exclusive; the parser calls `parser.error()` if both are supplied.

---

### Overall flow

Both execution modes share the same pipeline core. The difference is only in how and when data is extracted and how results are accumulated.

**Full mode (`run_inventory`)**

1. Optionally extract all `.zip` archives from `--zip-dir` into `--extract-dir` in a single pass.
2. Run `_run_pipeline_steps` once over the entire `--extract-dir`.
3. Compute missing cases (BP001–BP999 gaps) over the complete dataset.
4. Add class status and gender status, write gender back to `classes.csv`.
5. Write `inventory_summary.csv` and log a status breakdown.

**Partitioned mode (`run_partitioned`)**

1. For each `.zip` archive in sorted order:
   a. Delete and recreate `--extract-dir` to reclaim disk space.
   b. Extract the single archive.
   c. Run `_run_pipeline_steps` on the freshly extracted folder.
   d. Save an interim CSV to `output/inventory/interim/<stem>.csv`.
   e. Delete `--extract-dir` again.
2. After all archives are processed, merge all batch rows in memory.
3. Compute missing cases over the merged dataset.
4. Add class and gender status, write gender back to `classes.csv`.
5. Merge per-batch `mixed_series_analysis_<stem>.csv` files into one.
6. Write `inventory_summary.csv` and log a status breakdown.

The interim CSVs written in step 1d serve as crash recovery checkpoints: if the process is interrupted after several batches, the rows already collected are not lost (they are not automatically resumed on re-run, but the interim files remain on disk for manual inspection).

---

### What `_run_pipeline_steps` does

This private helper encapsulates the shared pipeline core called by both modes.

```
get_subfolders → organize_data → validate_dcms → analyze_mixed_folders → join_class_data
```

- `get_subfolders(extract_dir)` — lists immediate subdirectories using a single `os.scandir` call.
- `organize_data(case_folders, extract_dir)` — normalises raw folder names to `BP{NNN}`, classifies folder layouts as `READY / SUBFOLDER_PATH / DUPLICATE_DATA / EMPTY`, appends `MISSING` sentinel rows for every gap in the BP001–BP999 range (internal to the batch), and resolves the DICOM path for each case.
- `validate_dcms(organized_data, extract_dir)` — reads DICOM headers (pixel data skipped via `stop_before_pixels=True`), checks for mixed series UIDs, filters scout images, sorts slices by physical position projection, evaluates inter-slice spacing, records orientation/modality/dimensions/sex/exam\_size, and flags size outliers against fixed 90–300 mm limits.
- `analyze_mixed_folders(validated_data, extract_dir, mixed_csv_path)` — for any case flagged `MIXED_SERIES_ERROR`, produces a per-series breakdown CSV intended for manual review.
- `join_class_data(validated_data, classes_csv)` — looks up each case's label, anatomical location, and patient age from `classes.csv` and merges them into the record. Strips duplicate-disambiguating suffixes (`BP001A` → `BP001`) before the lookup.

`_run_pipeline_steps` then strips the `MISSING` sentinel rows from its return value. Those sentinels are meaningless within a partial batch (BP001–BP999 gaps computed inside a single zip would include every case not in that zip). The true missing-case computation is deferred to after all batches are merged.

Note: `check_missing_class` (from `class_utils`) is intentionally not called inside `_run_pipeline_steps`. Class-label presence is instead tracked by the separate `class_status` column populated by `add_class_status`, which runs once over the full dataset after merging. This avoids overwriting `validation_status` with `MISSING_CLASS` and losing diagnostic information for cases that failed DICOM validation.

---

### Function reference

#### `add_class_status(data, classes_csv, logger) -> list[dict]`

Adds a `class_status` field (`"OK"` or `"MISSING"`) to every record, including those whose `validation_status` is already `MISSING` or `NOT_APPLICABLE`. This is the key difference from `class_utils.check_missing_class`, which only examines cases with `validation_status == "OK"` and overwrites that field. `add_class_status` exists so that a case which is both absent from the filesystem and unlabelled appears as `MISSING` in both columns simultaneously without any column overwriting the other.

- **Parameters:** `data` — full list of case dicts (including MISSING sentinel rows); `classes_csv` — path to `classes.csv`; `logger` — logger instance.
- **Returns:** the same list, each dict extended with `"class_status"` (in-place).
- **Side effects:** reads `classes.csv`; logs the count of unlabelled cases.
- **Why this way:** decoupling class-label reporting from `validation_status` preserves diagnostic information for all failure modes simultaneously.

#### `check_missing_gender(data, logger) -> list[dict]`

Adds a `gender_status` field (`"OK"` or `"MISSING"`) to every record based on whether `patient_sex` equals `"M"` or `"F"`. Checks all cases regardless of validation outcome.

- **Parameters:** `data` — full list of case dicts; `logger` — logger instance.
- **Returns:** same list with `"gender_status"` added in-place.
- **Side effects:** logs the count of cases with missing sex tags.
- **Why this way:** symmetric with `add_class_status` — both status columns are populated independently so any combination of failures is visible.

#### `write_gender_to_classes_csv(data, classes_csv, logger) -> None`

Reads `classes.csv`, adds a `gender` column if absent, and writes back numeric gender values (M → 0, F → 1) for any case where the column was previously empty. Never overwrites an existing non-empty value. Strips duplicate suffixes (`BP001A` → `BP001`) before the CSV lookup, consistent with `class_utils` behaviour.

- **Parameters:** `data` — full list of case dicts after gender check; `classes_csv` — path to `classes.csv`; `logger` — logger instance.
- **Returns:** nothing.
- **Side effects:** modifies `classes.csv` on disk; logs the count of new values written.
- **Why this way:** mirrors `diagnostics/extract_gender.py` but runs inline during the inventory scan so gender data accumulates incrementally without a separate script invocation.

#### `_find_zips(zip_dir, logger) -> list[Path]`

Returns a sorted list of `.zip` files in `zip_dir`. Logs a warning if none are found.

- **Parameters:** `zip_dir` — source directory; `logger`.
- **Returns:** sorted list of `Path` objects.
- **Side effects:** logging only.

#### `_extract_zip(zip_path, extract_dir, logger) -> bool`

Extracts a single `.zip` archive into `extract_dir`. Returns `True` on success, `False` on `BadZipFile` or any other exception, logging the specific error. Does not raise.

- **Parameters:** `zip_path` — path to the archive; `extract_dir` — destination; `logger`.
- **Returns:** `True` (success) or `False` (failure).
- **Side effects:** creates files under `extract_dir`; logging.

#### `_run_pipeline_steps(extract_dir, mixed_csv_path, logger) -> list[dict]`

Runs the five-stage pipeline core and returns all records except MISSING sentinel rows. See "What `_run_pipeline_steps` does" above.

- **Parameters:** `extract_dir` — directory of extracted case folders; `mixed_csv_path` — destination for the mixed-series CSV; `logger`.
- **Returns:** list of case record dicts with `data_code != "MISSING"`.
- **Side effects:** may write `mixed_csv_path` to disk; extensive logging.

#### `_write_summary_csv(rows, path) -> None`

Writes the list of case dicts to a CSV at `path` using the fixed `SUMMARY_FIELDNAMES` column order. Extra keys in the dicts are silently ignored (`extrasaction="ignore"`).

- **Parameters:** `rows` — list of dicts; `path` — destination `Path`.
- **Returns:** nothing.
- **Side effects:** creates or overwrites the file at `path`.

#### `_merge_mixed_csvs(csv_paths, output_path) -> None`

Concatenates multiple mixed-series CSVs (one per batch in partitioned mode) into a single file, keeping only one header row. Skips paths that do not exist. Does nothing if no rows are found.

- **Parameters:** `csv_paths` — list of `Path` objects; `output_path` — destination.
- **Returns:** nothing.
- **Side effects:** creates or overwrites `output_path`.

#### `_log_status_breakdown(data, logger) -> None`

Logs a formatted summary table of `validation_status` counts plus total counts of `class_status == "MISSING"` and `gender_status == "MISSING"` to the logger at INFO level. Uses `collections.Counter`.

- **Parameters:** `data` — full merged case list; `logger`.
- **Returns:** nothing.
- **Side effects:** logging only.

#### `run_inventory(zip_dir, extract_dir, skip_extract, log_dir) -> None`

Full-mode entry point. Optionally extracts all zips, runs the pipeline once, merges MISSING cases, adds class/gender status, writes gender back to `classes.csv`, writes the final CSV, and logs the breakdown.

- **Parameters:** `zip_dir`, `extract_dir`, `skip_extract` (bool), `log_dir` (optional `Path`).
- **Returns:** nothing.
- **Side effects:** may create `extract_dir`; reads all DICOM headers under `extract_dir`; writes `inventory.log`, `inventory_summary.csv`, `mixed_series_analysis.csv`, and modifies `classes.csv`.
- **Raises:** `RuntimeError` if `zip_dir` does not exist (when extraction is not skipped) or if `extract_dir` does not exist (when `--skip-extract` is used).

#### `run_partitioned(zip_dir, extract_dir, log_dir) -> None`

Partitioned-mode entry point. Processes one archive at a time, wiping `extract_dir` before and after each batch to minimise peak disk usage. After all batches, merges rows, computes missing cases, adds class/gender status, merges mixed CSVs, and writes the final summary.

- **Parameters:** `zip_dir`, `extract_dir`, `log_dir` (optional `Path`).
- **Returns:** nothing.
- **Side effects:** repeatedly creates and destroys `extract_dir` via `shutil.rmtree`; writes interim CSVs under `output/inventory/interim/`; writes `inventory.log`, `inventory_summary.csv`, `mixed_series_analysis.csv`, and modifies `classes.csv`.
- **Raises:** `RuntimeError` if `zip_dir` does not exist.

---

### Output artefacts

| File | Location | Description |
|---|---|---|
| `inventory.log` | `--log-dir/inventory.log` | JSONL runtime log (DEBUG level) |
| `inventory_summary.csv` | `--log-dir/inventory_summary.csv` | One row per case, 17 columns (see `SUMMARY_FIELDNAMES`) |
| `mixed_series_analysis.csv` | `--log-dir/mixed_series_analysis.csv` | Per-series breakdown for `MIXED_SERIES_ERROR` cases |
| `interim/<stem>.csv` | `--log-dir/interim/` | Partitioned mode only: one CSV per processed archive |
| `interim/mixed_<stem>.csv` | `--log-dir/interim/` | Partitioned mode only: per-batch mixed-series CSVs |
| `classes.csv` | `data_engine/dataset/classes.csv` | Modified in-place: `gender` column backfilled |

The `inventory_summary.csv` columns are, in order: `original_name`, `fixed_name`, `data_path`, `total_dcms`, `validation_status`, `class_status`, `gender_status`, `duplicate_slice_count`, `scout_slice_count`, `orientation`, `modality`, `slice_thickness`, `patient_age`, `patient_sex`, `image_dimensions`, `class`, `location`, `exam_size`.

---

### What it scans

The script scans `.zip` archives under `--zip-dir`. Each archive is expected to unpack into a flat collection of case folders whose names follow the `bp{N}` / `BP_{N}` / `BP{NNN}` convention used throughout the data engine. There is no validation of the archive structure before extraction; malformed archives are caught by `_extract_zip` and skipped.

Within the extracted tree, the scan is shallow: DICOM files are expected at the top level of each case folder or one subfolder deep. The folder-layout classifier (`add_data_codes`) does not recurse further.

---

### Overlap with `data_cleaner.py`

The two scripts share the same five-stage pipeline core (`organize_data`, `validate_dcms`, `analyze_mixed_folders`, `join_class_data`, the MISSING-case computation) and the same `classes.csv` path constant.

The differences are:

| Aspect | `data_cleaner.py` | `inventory_scan.py` |
|---|---|---|
| Input | Directory of extracted DICOM folders | Directory of `.zip` archives (or pre-extracted folder) |
| Conversion | Yes — runs NIfTI conversion | No — stops after validation |
| Missing-class handling | `check_missing_class` sets `validation_status = MISSING_CLASS` | `add_class_status` adds a separate `class_status` column; `validation_status` is not touched |
| Gender | Not checked | `check_missing_gender` adds `gender_status`; `write_gender_to_classes_csv` backfills `classes.csv` |
| Audit log | `ingestion_summary.csv` (conversion outcomes) | `inventory_summary.csv` (pre-conversion quality audit) |
| Output CSV columns | 16 columns (no `class_status`, no `gender_status`) | 17 columns (adds `class_status` and `gender_status`) |
| Low-disk mode | Not supported | `--partitioned` mode processes one zip at a time |
| Mount-point guard | Yes — raises `RuntimeError` if output drive is not mounted | No |

`inventory_scan.py` is purely informational and makes no permanent changes to the filesystem except backfilling `classes.csv` with gender values and writing its own output CSVs. It does not rename folders, does not create NIfTI files, and does not move or delete source DICOM data.

---

### Complex control flow worth noting

**`_run_pipeline_steps` — MISSING row filtering.** The function strips sentinel rows whose `data_code == "MISSING"` from its return value, then re-injects real missing cases at the outer level after all batches are merged. This is correct but requires understanding two separate passes: an inner per-batch `find_missing_cases` call (inside `organize_data`) whose results are discarded, and an outer post-merge call whose results are kept. A comment in the source flags this explicitly, but the asymmetry is a potential maintenance trap.

**`run_partitioned` — repeated `shutil.rmtree`.** The loop deletes `extract_dir` both at the start of each iteration (to clean up from the previous iteration) and at the end (immediately after processing). If the loop body raises between the two `rmtree` calls, the directory is left on disk. This is intentional (the interim CSV has already been saved), but it means a failed run leaves data behind for manual inspection rather than cleaning up automatically.

**`add_data_codes` — nested condition tree.** The data-code assignment uses a two-level `if/elif/else` over `non_empty_subfolders` count and `direct_items` count. There are five possible outcomes from four branches. The logic is correct and the inline comments are clear, but it could be simplified to a lookup table or match statement on the pair `(direct_items > 0, non_empty_subfolders)` if the branch count grows.

**`_evaluate_spacing` — two-spacing branch.** The `len(unique_spacings) == 2` branch applies a `np.isclose` tolerance of `atol=0.1` mm to detect the "one missing slice" pattern. This tolerance is undocumented as a constant and is easy to overlook. If the typical slice thickness changes significantly (e.g. sub-millimetre isotropic acquisitions), this value may produce false positives.

---

### Notes on imported modules

The four imported modules are shared with `data_cleaner.py` and are not specific to `inventory_scan.py`.

- **`src.file_utils`** — `get_subfolders` and `organize_data` are used directly. `find_missing_cases` is imported separately so `inventory_scan.py` can call it after merging batches, outside the scope of `organize_data`. This bypasses the internal call that `organize_data` makes, which is discarded.
- **`src.dicom_utils`** — `validate_dcms` and `analyze_mixed_folders` are called identically to `data_cleaner.py`. The `_flag_exam_size_outliers` helper and all private functions are internal to the module and not re-exported.
- **`src.class_utils`** — Only `join_class_data` is imported. `check_missing_class` is deliberately not used; its role is replaced by `add_class_status`.
- **`src.logging_utils`** — Only `setup_logger` is imported. `write_audit_log` and `log_jsonl` are not used because `inventory_scan.py` writes its own CSV via `_write_summary_csv`.
