## Pipeline Overview

### End-to-End Data Flow: Raw DICOM → NIfTI Dataset on Disk

The data engine transforms raw hospital DICOM exports into a set of class-labelled, spatially normalised NIfTI volumes ready for CNN training. The full process runs in two sequential, independently invoked scripts:

1. **`data_engine/data_cleaner.py`** — DICOM ingestion and NIfTI conversion.
2. **`data_engine/dataset_gen.py`** — Spatial preprocessing into six variant datasets.

A third script, **`data_engine/inventory_scan.py`**, is a preflight audit tool that runs the same validation logic as `data_cleaner.py` but stops before conversion, producing only a quality report CSV.

---

### Script Execution Order

```
# Step 0 (optional preflight — run before committing to conversion):
python data_engine/inventory_scan.py \
    --zip-dir /mnt/data/raw \
    --extract-dir /mnt/data/raw_extract

# Step 1 — DICOM validation and NIfTI conversion:
python data_engine/data_cleaner.py \
    --raw-dir /mnt/data/cases-3/raw \
    --nifti-dir /mnt/data/cases-3/nifti \
    --workers 8

# Step 2 — Spatial preprocessing into variant datasets:
python data_engine/dataset_gen.py \
    --input-dir /mnt/data/cases-3/nifti \
    --output-base /mnt/data/cases-3 \
    --target-shape 192x192x128 \
    --variants C D A B E F
```

Steps 1 and 2 are always run in that order. Step 0 can be run before step 1 on any new batch of raw data to identify issues without writing any NIfTI files.

---

### Stage 1: `data_cleaner.py` — DICOM to NIfTI

`run_pipeline()` is the entry point. It calls nine sequential steps, threading all state through a shared `list[dict]` that grows one key at a time as each step adds information.

#### Sub-stage 1a: Folder Discovery (`file_utils.organize_data`)

`run_pipeline` calls `get_subfolders(raw_dir)` which returns `list[str]` — the raw directory names found under `--raw-dir`, e.g. `['bp1', 'BP_002', 'bp003']`.

These names are passed to `organize_data(case_folders, raw_dir)`, which chains seven internal steps and returns a `list[dict]`. Each dict at this stage contains:

| Key | Type | Example |
|---|---|---|
| `original_name` | `str` | `'bp1'` |
| `fixed_name` | `str` | `'BP001'` |
| `data_code` | `str` | `'READY'` / `'SUBFOLDER_PATH'` / `'DUPLICATE_DATA'` / `'EMPTY'` / `'MISSING'` |
| `data_path` | `str` | `'bp1'` (relative) or `'bp2/DICOM'` for subfolder layout |
| `total_dcms` | `int` | `312` |
| `direct_items` | `int` | `312` (removed after this stage) |
| `items_in_subfolders` | `int` | `0` (removed after this stage) |

Cases in the BP001–BP999 range that have no folder on disk are appended with `data_code = 'MISSING'` and `original_name = 'missing'` by `find_missing_cases()` so the audit log is complete.

#### Sub-stage 1b: DICOM Validation (`dicom_utils.validate_dcms`)

`validate_dcms(organized_data, raw_dir)` iterates over the list, reads DICOM headers via `load_dicom_metadata()` (pixel data skipped, `stop_before_pixels=True`), and runs a ten-step validation on each series. Each dict is mutated in-place. New keys added:

| Key | Type | Example |
|---|---|---|
| `validation_status` | `str` | `'OK'` / `'MIXED_SERIES_ERROR'` / `'VARIABLE_SPACING'` / `'GAPPED_SEQUENCE'` / `'DUPLICATE_SLICES'` / `'BELOW_LIMIT'` / `'ABOVE_LIMIT'` / `'VALIDATION_ERROR: <msg>'` |
| `orientation` | `str` | `'AXIAL'` |
| `modality` | `str` | `'CT'` |
| `slice_thickness` | `float` or `str` | `0.625` |
| `image_dimensions` | `str` | `'512x512x312'` |
| `patient_sex` | `str` | `'M'` |
| `duplicate_slice_count` | `int` | `0` |
| `scout_slice_count` | `int` | `4` |
| `exam_size` | `float` or `str` | `195.0` (mm) |

`_flag_exam_size_outliers()` then reclassifies `OK` cases outside the 90–300 mm anatomical bounds to `BELOW_LIMIT` or `ABOVE_LIMIT`.

For any case flagged `MIXED_SERIES_ERROR`, `analyze_mixed_folders()` is called as a side-output step: it writes `{log_dir}/mixed_series_analysis.csv` but does not change the list.

#### Sub-stage 1c: Label Join (`class_utils.join_class_data`)

`join_class_data(validated_data, 'data_engine/dataset/classes.csv')` reads `classes.csv` (columns: `exam`, `class`, `location`, `Age`) into a dict keyed by uppercase exam name and merges three new fields into each matching record:

| Key | Type | Example |
|---|---|---|
| `class` | `str` | `'0'` or `'1'` |
| `location` | `str` | `'ACI D'` (blank for class 0) |
| `patient_age` | `str` | `'65'` |

Duplicate-disambiguated names (`BP001A`) are stripped to their base (`BP001`) before lookup. Records with no CSV match receive no new keys.

`check_missing_class()` then sets `validation_status = 'MISSING_CLASS'` for any record that was `OK` but has no `class` key.

#### Sub-stage 1d: Filtering and Conversion (`nifti_utils`)

`filter_for_conversion(final_data)` applies two gates and returns a filtered `list[dict]` — only records where `validation_status == 'OK'` and `class in ('0', '1')`.

`process_and_convert_exams(eligible_exams, raw_dir, nifti_dir)` dispatches each eligible exam to a `ProcessPoolExecutor`. Each worker runs `_convert_exam_worker()`, which:

1. Constructs the output path as `{nifti_dir}/{class}/{fixed_name}.nii.gz`.
2. Skips if the file already exists (idempotency).
3. Calls `convert_series_to_nifti(dicom_dir, output_path)`, which uses MONAI + ITKReader to stack slices into a 3D volume, casts to float32, and saves a compressed NIfTI with the affine matrix.

Each worker returns a result dict with keys `exam_name`, `status` (`'success'` / `'skipped'` / `'failed'`), `reason`, `output_path`.

#### Sub-stage 1e: Audit Outputs

`_build_audit_log(conversion_results, final_data)` merges the conversion results with synthesised `'failed'` entries for cases that never reached conversion, producing a `list[dict]` that covers every input case.

The following files are written to `{log_dir}/` (default: `data_engine/output/`):

| File | Written by | Contents |
|---|---|---|
| `ingestion.log` | `_JsonlHandler` (incremental) | JSONL runtime log, all DEBUG+ events |
| `mixed_series_analysis.csv` | `analyze_mixed_folders` | Per-series breakdown for `MIXED_SERIES_ERROR` cases; created only when such cases exist |
| `ingestion_summary.csv` | `write_audit_log` | One row per input case: `exam_name`, `status`, `reason`, `output_path` |
| `folder_rename_map.csv` | `run_pipeline` (inline `csv.DictWriter`) | One row per case: name mapping, DICOM metadata, classification fields, validation status (16 columns) |

And the primary output files written to `{nifti_dir}/`:

```
nifti/
├── 0/
│   ├── BP001.nii.gz    ← healthy cases (class 0)
│   ├── BP004.nii.gz
│   └── ...
└── 1/
    ├── BP002.nii.gz    ← aneurysm cases (class 1)
    ├── BP003.nii.gz
    └── ...
```

---

### Stage 2: `dataset_gen.py` — Spatial Preprocessing Variants

`main()` is the entry point. It collects all `.nii.gz` files from `{nifti_dir}/*/*.nii.gz` (preserving class subdirectory structure), shuffles them, then calls `_run_variant()` once per requested variant key in order.

`_run_variant(variant_key, nifti_files, n_workers, target_shape, target_shape_key, output_base)` creates a fresh fork-based `multiprocessing.Pool` with `maxtasksperchild=1`, dispatches one task per file via `imap_unordered`, and returns a list of `(label, error)` pairs for failures. Each task function (`_task_a` through `_task_f`) calls `_load(file_path)` to read the source NIfTI and then calls the corresponding `_variant_X_from_data(data, affine, output_path, target_shape)` from `nifti_resize.py`.

The six variants implement a 3×2 factorial design (three resampling strategies × two crop strategies):

| | 80% centre crop | Full volume |
|---|---|---|
| No resampling | C | D |
| Dynamic isotropic resample | A | B |
| Fixed isotropic resample | E | F |

The data flow inside each `_variant_X_from_data` call is:

- **A:** `_center_crop` → `_dynamic_spacing` (on cropped volume) → `_resample_isotropic` → `ResizeWithPadOrCrop` (zero-pad) → `_save`
- **B:** `_dynamic_spacing` (full volume) → `_resample_isotropic` → `ResizeWithPadOrCrop` → `_save`
- **C:** `_center_crop` → MONAI `Resize` (trilinear) → affine column-scale → `_save`
- **D:** MONAI `Resize` (trilinear) → affine column-scale → `_save`
- **E:** `_center_crop` → `_resample_isotropic` (fixed spacing from `E_SPACING_MM_BY_SHAPE`) → `ResizeWithPadOrCrop` → `_save`
- **F:** `_resample_isotropic` (fixed spacing from `F_SPACING_MM_BY_SHAPE`) → `ResizeWithPadOrCrop` → `_save`

`_save()` asserts `data.shape == target_shape` before writing, surfacing shape bugs immediately rather than letting wrongly-sized files corrupt dataset batches downstream.

Output files are written to `{output_base}/{folder_name}/{class}/{filename}.nii.gz` where `folder_name` is looked up from `OUTPUT_FOLDER_NAMES[target_shape_key][variant_key]`:

```
/mnt/data/cases-3/
├── nifti/                     ← Stage 1 output / Stage 2 input
│   ├── 0/BP001.nii.gz
│   └── 1/BP002.nii.gz
├── dataset_A192/              ← variant A, 192×192×128
│   ├── 0/BP001.nii.gz
│   └── 1/BP002.nii.gz
├── dataset_B192/
├── dataset_C192/
├── dataset_D192/
├── dataset_E192/
├── dataset_F192/
├── dataset_A176/              ← same variants at 256×256×176
│   ...
└── dataset_A_resampled_cropped/  ← legacy 128×128×128 names
    ...
```

Filenames are preserved exactly from the `nifti/` input, so `BP001.nii.gz` appears identically in every dataset subdirectory.

---

### `inventory_scan.py`: Preflight vs. Production Run

`inventory_scan.py` and `data_cleaner.py` share the same five-stage pipeline core:

```
get_subfolders → organize_data → validate_dcms → analyze_mixed_folders → join_class_data
```

The critical difference is what happens after that shared core:

| Aspect | `inventory_scan.py` (preflight) | `data_cleaner.py` (production) |
|---|---|---|
| Input | `.zip` archives (extracted inline) or pre-extracted directory | Directory of DICOM case folders already on disk |
| NIfTI conversion | None — stops after validation | Yes — runs `filter_for_conversion` + `process_and_convert_exams` |
| Missing-class handling | `add_class_status` adds a separate `class_status` column; `validation_status` is never touched | `check_missing_class` sets `validation_status = 'MISSING_CLASS'` for unlabelled OK cases |
| Gender | `check_missing_gender` adds `gender_status`; `write_gender_to_classes_csv` backfills `classes.csv` | Not checked |
| Primary output | `inventory_summary.csv` (17 columns, pre-conversion quality audit) | `folder_rename_map.csv` + `ingestion_summary.csv` (conversion outcomes) |
| Low-disk mode | `--partitioned` processes one zip at a time, deleting extracted files after each batch | Not supported |
| Mount-point guard | None | Raises `RuntimeError` if `--nifti-dir` is not on a mounted filesystem |
| Side effects on source data | None (reads only; writes only its own output CSVs and backfills `classes.csv` gender column) | Writes `.nii.gz` files to `--nifti-dir` |

**Intended workflow:** Run `inventory_scan.py` first on a new batch of zipped DICOM exports to check coverage, validation rates, and label completeness. Once the inventory report shows an acceptable number of `OK` cases with labels, extract the data properly and run `data_cleaner.py` for the actual conversion.

The key reason `inventory_scan.py` does not call `check_missing_class` is to preserve `validation_status` as a clean record of the DICOM quality outcome. Instead it uses a separate `class_status` column so that a case which failed DICOM validation AND has no label shows both problems simultaneously — `validation_status = 'VARIABLE_SPACING'` and `class_status = 'MISSING'` — without either column overwriting the other.

---

### Diagnostics Scripts: Role in the Pipeline

The scripts under `data_engine/diagnostics/` are standalone analysis tools, not part of the automated pipeline. They are run manually, on an as-needed basis, and they only read data — they never write NIfTI files or alter DICOM source data (except `extract_gender.py`, which backfills `classes.csv`).

**When they are run:** After `data_cleaner.py` has produced the `nifti/` directory, and optionally after `dataset_gen.py` has produced the variant datasets. They are not required for the pipeline to function; they exist to characterise the dataset and validate preprocessing choices.

| Script | Input | Output | Purpose |
|---|---|---|---|
| `survey_nifti.py` | `{nifti_dir}/*/*.nii.gz` | `diagnostics/outputs/nifti_survey.csv` | Per-exam dimensions, voxel spacing, physical FOV, isotropy flag |
| `analyse_nifti_survey.py` | `diagnostics/outputs/nifti_survey.csv` | Console summary + `diagnostics/outputs/survey_plots/*.png` | Distribution analysis across classes; simulation of padding/cropping waste for candidate target shapes |
| `check_fov_air.py` | `{nifti_dir}/*/*.nii.gz` + `nifti_survey.csv` + `nifti_survey_groups.csv` | `diagnostics/outputs/fov_air.csv` | Measures how many millimetres of air border surround the actual anatomy on each face, exposing how much of the stated FOV is empty space |
| `check_resize_quality.py` | `nifti_survey.csv` + all `dataset_*/` directories | `diagnostics/outputs/resize_quality.csv` | Computes pad percentage and crop percentage for each exam across every variant dataset, verifying the resampling strategy works as intended |
| `check_nifti.py` | Single `.nii.gz` file (CLI argument) | Console only | Spot-check: shape, dtype, intensity range, non-zero voxel count |
| `check_nifti_quality.py` | Single `.nii.gz` file (hardcoded path) | Console + `/tmp/nifti_slice_<name>.png` | Deeper header inspection: affine matrix, NaN/Inf count, mid-slice visualisation |
| `compute_optimal_spacing.py` | Hardcoded dataset group statistics | Console | Computes the weighted-optimal fixed isotropic spacing for variant E/F by maximising a padding/cropping efficiency metric across the four scanner archetypes |
| `extract_gender.py` | `.zip` archives under `/mnt/data/raw` | Backfills `classes.csv` with `gender` column | One-time script that reads `PatientSex` DICOM tags from zipped raw data and writes numeric gender values into `classes.csv`; superseded by `write_gender_to_classes_csv` in `inventory_scan.py` |
| `check_missing_gender.py` | `classes.csv` | Console | Reports cases in `classes.csv` that have no `gender` entry |

**Data consumed by diagnostics:** All diagnostics scripts read from the `nifti/` directory produced by `data_cleaner.py` and/or from the `dataset_*/` directories produced by `dataset_gen.py`. Several scripts (`check_fov_air.py`, `check_resize_quality.py`) depend on `nifti_survey.csv` produced by `survey_nifti.py`, so `survey_nifti.py` should be run first.

**Data produced by diagnostics:** The diagnostics write CSV files and plots to `data_engine/diagnostics/outputs/`. These outputs are consumed only by other diagnostic scripts or by the human reviewer, never by the main pipeline. The only exception is `extract_gender.py` and `write_gender_to_classes_csv` (inside `inventory_scan.py`), which modify `data_engine/dataset/classes.csv` — the only source of ground-truth labels.

**Relation to pipeline design decisions:** `compute_optimal_spacing.py` and `analyse_nifti_survey.py` informed the choice of the 192×192×128 and 256×256×176 target shapes and the fixed spacing values in `E_SPACING_MM_BY_SHAPE` / `F_SPACING_MM_BY_SHAPE` in `nifti_resize.py`. They are not re-run on every pipeline execution; they are research tools used to derive constants that are then hardcoded.

---

### Module Dependency Map

```
data_cleaner.py (entry point)
├── src/file_utils.py      get_subfolders, organize_data
├── src/dicom_utils.py     validate_dcms, analyze_mixed_folders
├── src/class_utils.py     join_class_data, check_missing_class
├── src/nifti_utils.py     filter_for_conversion, process_and_convert_exams
└── src/logging_utils.py   setup_logger, write_audit_log

inventory_scan.py (entry point — preflight only)
├── src/file_utils.py      get_subfolders, organize_data, find_missing_cases
├── src/dicom_utils.py     validate_dcms, analyze_mixed_folders
├── src/class_utils.py     join_class_data  (check_missing_class NOT used)
└── src/logging_utils.py   setup_logger  (write_audit_log NOT used)

dataset_gen.py (entry point — runs after data_cleaner.py)
└── src/nifti_resize.py    _load, _variant_a_from_data … _variant_f_from_data,
                           VALID_TARGET_SHAPES

diagnostics/* (standalone — run after data_cleaner.py and/or dataset_gen.py)
    No imports from src/; read nifti/ and dataset_*/ directly via nibabel/numpy.
```

---

### Data Structure at Each Pipeline Stage

| After stage | Type | Key fields present |
|---|---|---|
| `get_subfolders` | `list[str]` | raw folder names |
| `organize_data` | `list[dict]` | `original_name`, `fixed_name`, `data_code`, `data_path`, `total_dcms` |
| `validate_dcms` | `list[dict]` | + `validation_status`, `orientation`, `modality`, `slice_thickness`, `image_dimensions`, `patient_sex`, `duplicate_slice_count`, `scout_slice_count`, `exam_size` |
| `join_class_data` | `list[dict]` | + `class`, `location`, `patient_age` |
| `check_missing_class` | `list[dict]` | `validation_status` updated to `'MISSING_CLASS'` for unlabelled OK cases |
| `filter_for_conversion` | `list[dict]` | subset: only `validation_status == 'OK'` and `class in ('0', '1')` |
| `process_and_convert_exams` | `list[dict]` | `exam_name`, `status`, `reason`, `output_path` (conversion result per case) |
| `_build_audit_log` | `list[dict]` | same schema, extended with synthesised entries for pre-conversion rejections |

## data_engine Needs Assessment

### 1. CSV Fragmentation — the Single-View Problem

**Current state.** A pipeline run produces four separate CSVs covering the same set of cases:

| File | Written by | Rows | Key columns |
|---|---|---|---|
| `folder_rename_map.csv` | `run_pipeline` step 9 (inline `csv.DictWriter`) | One per input case | `original_name`, `fixed_name`, `data_path`, `total_dcms`, `validation_status`, `duplicate_slice_count`, `scout_slice_count`, `orientation`, `modality`, `slice_thickness`, `patient_age`, `patient_sex`, `image_dimensions`, `class`, `location`, `exam_size` |
| `ingestion_summary.csv` | `logging_utils.write_audit_log` | One per input case | `exam_name`, `status`, `reason`, `output_path` |
| `mixed_series_analysis.csv` | `dicom_utils.analyze_mixed_folders` | One per series within a mixed folder | (series-level breakdown) |
| `inventory_summary.csv` | `inventory_scan._write_summary_csv` | One per expected case | 17 columns including `class_status`, `gender_status` |

To answer "which exams are OK and why were the others excluded, and what was the conversion outcome?" a user must open two files and perform a manual join on `fixed_name` / `exam_name`.

**Proposed merge: `pipeline_report.csv`.**

Join `folder_rename_map.csv` and `ingestion_summary.csv` in `_build_audit_log`, which already iterates both datasets in `data_cleaner.py`. At the point where `_build_audit_log` constructs the synthesised `failed` rows from `final_data`, it already has access to all `folder_rename_map` columns. The join key is `fixed_name` (from `final_data`) = `exam_name` (from `conversion_results`).

Concrete column set for `pipeline_report.csv` (drop `ingestion_summary.csv` entirely; replace the step-9 `DictWriter` with a single write):

```
original_name, fixed_name, data_path, total_dcms,
validation_status, duplicate_slice_count, scout_slice_count,
orientation, modality, slice_thickness, patient_age, patient_sex,
image_dimensions, class, location, exam_size,
conversion_status, conversion_reason, output_path
```

The last three columns (`conversion_status`, `conversion_reason`, `output_path`) come from `conversion_results` (currently written as `status`, `reason`, `output_path` in `ingestion_summary.csv`). Cases that never reached conversion — those synthesised by `_build_audit_log` — get `conversion_status = "not_attempted"`, `conversion_reason` = the `validation_status` string, and `output_path = ""`.

**Changes required:**
- `data_cleaner.run_pipeline`: replace the separate `write_audit_log` call (step 8) and inline `DictWriter` block (step 9) with a single function call — `write_pipeline_report(full_audit_results, final_data, path)` — in `logging_utils.py`.
- `logging_utils`: delete `write_audit_log`; add `write_pipeline_report` that performs the join and writes the merged CSV.
- `_build_audit_log`: can be absorbed into `write_pipeline_report` since it only exists to feed `write_audit_log`.
- `mixed_series_analysis.csv` stays separate — it is per-series, not per-case, and is already clearly scoped.

---

### 2. Logging — JSONL Log Is Unused

**Problems identified.**

1. `_JsonlHandler` in `logging_utils.py` writes `ingestion.log` in JSONL format. No script in the repository reads or processes this file. The module docstring explicitly justifies JSONL for "post-run analysis" but that analysis never materialises.

2. `log_jsonl` (lines 193–215 of `logging_utils.py`) is dead code. Its own docstring says "This function is a standalone utility and is not called by the main pipeline." It mutates its `data` argument in place via `data['timestamp'] = ...`, which is a side-effect bug.

3. `write_audit_log` is a thin CSV wrapper whose only caller is `run_pipeline` step 8. If the pipeline report is merged (see §1), this function is also removed.

4. The custom `_JsonlHandler` class is 30 lines of code that duplicate what `logging.FileHandler` plus a one-line `Formatter` provides. Replacing it with stdlib requires only:

```python
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'
))
```

**Proposed changes.**
- Delete `_JsonlHandler`. Replace with a stdlib `FileHandler` and a `Formatter` that includes timestamp, level, logger name, and message. This eliminates `import json` from `logging_utils.py`.
- Delete `log_jsonl` entirely.
- Delete `write_audit_log` once the pipeline report merge (§1) is implemented.
- Remove `from datetime import datetime` from `logging_utils.py` (only used by `_JsonlHandler.emit` and `log_jsonl`).
- Keep `setup_logger` and its duplicate-handler guard unchanged — both are justified.

The net result is `logging_utils.py` shrinks from ~215 lines to ~30 lines: one `setup_logger` function and one `write_pipeline_report` function.

---

### 3. NIfTI Conversion vs. Quality Checking Separation

**Current state in `nifti_utils.py`.**

`nifti_utils.py` contains no quality-checking logic — all QA is upstream in `dicom_utils.validate_dcms`. The module is already cleanly scoped. No split is needed within `nifti_utils.py` itself.

`dicom_utils.py` is where the mixing occurs. Two functions contain embedded quality/validation logic that is partly duplicated in `analyze_mixed_series`:

- `validate_dcms` — runs conversion-prerequisite checks (scout filtering, UID multiplicity, spatial metadata presence) alongside pure geometric analysis (`_sort_slices_by_projection`, `_evaluate_spacing`). The scout-filtering predicate `'ORIGINAL' in ImageType`, the UID-grouping check, and the `isinstance(effective_slice_thickness, float)` guard are all quality checks, not I/O.
- `analyze_mixed_series` — duplicates the scout filter, `get_orientation`, `_sort_slices_by_projection`, and `_evaluate_spacing` calls verbatim from `validate_dcms` (flagged in the existing doc at line 1385).

**Proposed split if the module is ever decomposed.**

| Target module | Functions |
|---|---|
| `nifti_io.py` | `convert_series_to_nifti`, `_convert_exam_worker`, `process_and_convert_exams` — pure DICOM-to-NIfTI I/O and parallelisation |
| `nifti_qa.py` | `filter_for_conversion` — reads QA results to gate I/O |
| `dicom_qa.py` (rename of `dicom_utils.py`) | `validate_dcms`, `_flag_exam_size_outliers`, `_evaluate_spacing`, `_sort_slices_by_projection`, `get_orientation`, `load_dicom_metadata` — all QA |
| `dicom_series.py` | `analyze_mixed_series`, `analyze_mixed_folders` — series inspection and CSV reporting |

The most impactful concrete action, independent of any rename, is to extract the shared series-validation logic into `_validate_single_series(metadata_list, iop) -> dict` and call it from both `validate_dcms` and `analyze_mixed_series`, eliminating the current verbatim duplication.

---

### 4. Diagnostics Consolidation

**Current directory contents** (`data_engine/diagnostics/`, 9 scripts):

| Script | What it does | Input required |
|---|---|---|
| `survey_nifti.py` | Scans NIfTI dir, writes `nifti_survey.csv` with per-exam shape/spacing/FOV | NIfTI directory |
| `analyse_nifti_survey.py` | Reads `nifti_survey.csv`, prints stats, saves plots to `survey_plots/` | `nifti_survey.csv` |
| `check_nifti.py` | Spot-checks one NIfTI file (shape, dtype, value range) — 14 lines, no argparse | One `.nii.gz` path (argv[1]) |
| `check_nifti_quality.py` | Detailed single-file check: header, affine, value distribution, optional DICOM comparison | One `.nii.gz` path, hardcoded `NIFTI_DIR` |
| `check_resize_quality.py` | Measures pad/crop percentages and spacing across all 8 variant datasets | `nifti_survey_variants.csv`, `nifti_survey.csv`, variant dirs |
| `check_fov_air.py` | Measures air-border depth on all 6 faces of every NIfTI exam | NIfTI dir, `nifti_survey.csv`, `nifti_survey_groups.csv` |
| `compute_optimal_spacing.py` | Computes optimal fixed isotropic spacing for variant E/F by weighted efficiency | `nifti_survey.csv` |
| `check_missing_gender.py` | Cross-references `nifti_survey.csv` against `classes.csv`, prints exams with missing gender | `classes.csv`, `nifti_survey.csv` — uses `pandas` |
| `extract_gender.py` | Reads PatientSex from DICOM headers in zip files, writes to `classes.csv` | `/mnt/data/raw` zip archives |

**Scripts that can be merged into a single `diagnose.py` entry point with subcommands:**

`survey_nifti.py`, `analyse_nifti_survey.py`, `check_nifti.py`, `check_nifti_quality.py`, `check_resize_quality.py`, and `check_fov_air.py` all operate on the NIfTI output directory and share the same runtime dependency set (`nibabel`, `numpy`, optional `matplotlib`). They form a natural group under a `python diagnose.py <subcommand>` interface:

```
python diagnose.py survey       # → survey_nifti.py
python diagnose.py analyse      # → analyse_nifti_survey.py
python diagnose.py check FILE   # → check_nifti.py (spot check)
python diagnose.py quality FILE # → check_nifti_quality.py (detailed)
python diagnose.py resize       # → check_resize_quality.py
python diagnose.py fov-air      # → check_fov_air.py
```

`compute_optimal_spacing.py` belongs in this group too (reads `nifti_survey.csv`, same deps).

**Scripts that should stay separate:**

- `extract_gender.py` — operates on raw DICOM zip archives, requires `pydicom`, has destructive side effects on `classes.csv`, and is conceptually part of the ingestion pipeline rather than NIfTI diagnostics. Its functionality has already been absorbed into `inventory_scan.write_gender_to_classes_csv`, making it redundant (see §5 overlap discussion).
- `check_missing_gender.py` — 12-line throwaway script that uses `pandas` for a cross-reference that `inventory_scan._log_status_breakdown` now produces automatically. It should be deleted, not preserved.

---

### 5. `inventory_scan.py` vs. `data_cleaner.py` Overlap

**Documented overlap.** Both scripts call the identical five-function pipeline core:

```
get_subfolders → organize_data → validate_dcms → analyze_mixed_folders → join_class_data
```

Both write a `mixed_series_analysis.csv`. Both compute MISSING cases against the BP001–BP999 range. Both resolve `CLASSES_CSV_PATH` from the same constant. The existing doc section "Overlap with `data_cleaner.py`" (line 1919) enumerates the differences accurately.

**Additional overlap not yet documented:**

- `inventory_scan.write_gender_to_classes_csv` duplicates the logic of `diagnostics/extract_gender.py`, as the `inventory_scan.py` docstring itself notes.
- `inventory_scan.add_class_status` performs the same CSV lookup as `class_utils.join_class_data` + `check_missing_class`, but writes to a separate column rather than overwriting `validation_status`. The lookup code (open CSV, build dict keyed by uppercased exam name, regex-strip suffix) is duplicated verbatim.

**Recommendation: keep `inventory_scan.py` separate but formalise its relationship to `data_cleaner.py`.**

Do not merge the two scripts and do not eliminate `inventory_scan.py`. The distinction is genuinely useful: `inventory_scan.py` is a preflight audit tool that operates on raw zip archives without committing to disk-intensive NIfTI conversion, while `data_cleaner.py` is the production pipeline that converts data.

What should change:

1. Rename `inventory_scan.py`'s entry points to make the relationship explicit in the CLI:
   - `run_inventory` → `run_preflight` or keep as-is with a clearer module docstring.

2. Eliminate the duplicated CSV-lookup code in `add_class_status` by delegating to `class_utils.join_class_data` for the actual lookup, then adding the `class_status` column in a post-join pass — the existing `join_class_data` already builds the dict and applies the regex strip.

3. Delete `diagnostics/extract_gender.py` — it is fully superseded by `inventory_scan.write_gender_to_classes_csv`.

4. Delete `diagnostics/check_missing_gender.py` — its output is produced automatically by `inventory_scan._log_status_breakdown`.

---

### 6. Long Comments

The following locations contain paragraph-length docstrings or inline comments that explain what the code already makes self-evident. These should be reduced to a single sentence or removed.

**`nifti_utils.py` — `convert_series_to_nifti` docstring (lines 111–171).**
The docstring is 60 lines. The sections "What is the affine matrix?" (8 lines of prose explaining what an affine is) and "Float32 casting:" (5 lines) explain medical imaging concepts that are not specific to the function's behaviour. A reader who does not know what an affine is needs domain education, not inline prose. Reduce to: the four bullet-point design choices (ITKReader, float32, affine preservation, explicit GC) expressed as one sentence each.

**`nifti_utils.py` — module-level docstring (lines 1–52).**
The "What is NIfTI and why convert to it?" section (lines 14–20) and "What is the affine matrix?" prose belong in a project README or wiki, not the module docstring. The module docstring should state what the module does and list its public API — both of which it also does (lines 49–51). Cut the background exposition; keep the public API table.

**`logging_utils.py` — `setup_logger` docstring (lines 96–144).**
The paragraph starting "The guard `if logger.handlers: return logger` prevents duplicate handlers" (8 lines) explains a 2-line guard that is clear from reading the code. Reduce to one sentence: "Guards against re-adding handlers if called more than once in the same process (e.g. in tests)."

**`logging_utils.py` — `_JsonlHandler` docstring (lines 52–71).**
"Extends Python's standard `FileHandler`, overriding only `emit()` to change the output format from plain text to JSON. Every other behaviour — file opening, rotation, error handling — is inherited." This is true of every `FileHandler` subclass and adds no information beyond what the class signature already communicates. If the class is kept, the docstring can be one sentence.

**`logging_utils.py` — module-level docstring (lines 2–38).**
The "Two output formats, two audiences" section (18 lines) and "Pipeline audit CSV" section (12 lines) each explain the design in prose that duplicates the public API table at the bottom. Since `log_jsonl` is dead code and `write_audit_log` is proposed for removal (§2), most of this prose becomes stale on the next refactor. Replace with a three-sentence summary.

**`data_cleaner.py` — inline comments in `run_pipeline` (lines 103–116).**
The mount-guard block has a 7-line inline comment explaining why the root-mount check exists. The `RuntimeError` message already contains a user-facing explanation. The comment can be cut to one line: `# Prevent silent writes to local disk when the external SSD is unmounted.`

**`inventory_scan.py` — `_run_pipeline_steps` docstring (lines 252–284).**
The note "Does NOT include MISSING sentinel rows — those are injected by `organize_data` internally but are meaningless when processing a partial batch" is repeated in a comment at line 279 and again in the "What `_run_pipeline_steps` does" section of the existing code doc. Pick one location.

---

### 7. Library Usage

**Third-party libraries used across `data_engine/src/` and `data_engine/diagnostics/`:**

| Library | Used in | Role | Replaceable? |
|---|---|---|---|
| `pydicom` | `dicom_utils.py`, `diagnostics/check_nifti_quality.py`, `diagnostics/extract_gender.py` | Reading DICOM headers | No — the only mature stdlib-free DICOM parser in Python |
| `numpy` | `dicom_utils.py`, `nifti_utils.py`, `nifti_resize.py`, most diagnostics | Vector math, array ops | No — MONAI and nibabel both return numpy arrays; removing numpy would require replacing the entire stack |
| `nibabel` | `nifti_utils.py`, `nifti_resize.py`, most diagnostics | Reading/writing NIfTI files | No — the standard NIfTI I/O library; SimpleITK is an alternative but is heavier |
| `monai` (transforms, data) | `nifti_utils.py`, `nifti_resize.py` | `LoadImage`, `ITKReader`, `Spacing`, `Resize`, `ResizeWithPadOrCrop`, `MetaTensor` | Partially — `Resize` and `ResizeWithPadOrCrop` could be replaced with `scipy.ndimage.zoom` + manual slicing; `Spacing` via `MetaTensor` is harder to replace without manual affine bookkeeping; `ITKReader` in `convert_series_to_nifti` could be replaced by `SimpleITK` but not stdlib |
| `torch` | `nifti_resize.py` (via `monai.data.MetaTensor`) | Tensor backend for MONAI | No — implicit MONAI dependency |
| `matplotlib` | `diagnostics/analyse_nifti_survey.py` | Survey plots | Yes, for the diagnostics context: `matplotlib` is not used in any pipeline module, only in one diagnostic script. If plots are removed the dependency disappears |
| `pandas` | `diagnostics/check_missing_gender.py` | CSV cross-reference | Yes — `diagnostics/check_missing_gender.py` is a 12-line script using `pd.read_csv` and `DataFrame.isin`. It should be deleted (§4); if kept, rewrite with stdlib `csv` + set intersection |

**Flags:**

- `import json` in `logging_utils.py` is used only by `_JsonlHandler.emit` and `log_jsonl`. Both are proposed for removal (§2). If they are removed, `json` and `from datetime import datetime` both become unused imports in that file.
- `from typing import Dict, Tuple` in `nifti_resize.py` — these are the pre-3.9 generic aliases. The codebase already uses `list[dict]`, `str | Path`, and other 3.10+ syntax elsewhere. Replace `Dict` with `dict` and `Tuple` with `tuple` throughout `nifti_resize.py` to be consistent.
- `import gc` in `nifti_utils.py` and `dataset_gen.py` — stdlib, but flagged because its presence is non-obvious. The existing docstrings explain it correctly (C-extension heap not visible to Python GC). No change needed; just confirming it is not a candidate for removal.
- `re` is imported in `inventory_scan.py`, `file_utils.py`, and `class_utils.py` for the same `BP\d+` stripping pattern. All three define the pattern inline. A single compiled `_BP_BASE_RE = re.compile(r"(BP\d+)")` in `file_utils.py` and imported where needed would centralise the domain-specific constant.

## Architectural Suggestions

### 1. NIfTI Conversion vs. Quality Checking Split

**Where the mixing actually occurs.** `nifti_utils.py` itself is already clean — all QA is upstream. The mixing problem is in `dicom_utils.py`, and specifically in the duplication between `validate_dcms` and `analyze_mixed_series`.

**Functions to move / rename:**

| Target module | Functions | Rationale |
|---|---|---|
| `nifti_io.py` (rename of `nifti_utils.py`) | `convert_series_to_nifti`, `_convert_exam_worker`, `process_and_convert_exams` | Pure DICOM-to-NIfTI I/O and subprocess orchestration; no QA logic. |
| `nifti_qa.py` (new, thin) | `filter_for_conversion` | Reads upstream QA results to gate I/O; it is the bridge between QA and conversion, not I/O itself. |
| `dicom_utils.py` (keep name, but scope it) | `load_dicom_metadata`, `_sort_slices_by_projection`, `_evaluate_spacing`, `get_orientation`, `validate_dcms`, `_flag_exam_size_outliers`, `analyze_mixed_series`, `analyze_mixed_folders` | All of these are quality checks or series inspection — none write NIfTI. No move required unless the module grows; the current name already fits. A `dicom_qa.py` rename would only add clarity if `dicom_utils.py` were later split to contain both pure I/O helpers and QA, which it currently does not. |

The single most impactful concrete action — independent of any file rename — is to extract the shared series-validation logic that `validate_dcms` and `analyze_mixed_series` both implement verbatim:

```python
# new private helper in dicom_utils.py
def _validate_single_series(metadata_list: list, iop: list) -> dict:
    """Sort, evaluate spacing, classify one series. Called by both validate_dcms
    and analyze_mixed_series."""
    _sort_slices_by_projection(metadata_list, iop)
    status, unique_spacings, dup_count, deltas = _evaluate_spacing(metadata_list)
    return {"validation_status": status, "unique_spacings": unique_spacings, ...}
```

Both callers currently duplicate: the scout-filter predicate (`'ORIGINAL' in ImageType`), the `get_orientation` call, `_sort_slices_by_projection`, and `_evaluate_spacing`. Extracting this into `_validate_single_series` eliminates the duplication without any cross-file restructuring.

**`dicom_qa.py` is only warranted** if a new module needs pure DICOM I/O helpers (e.g. a future writer) that should not depend on the QA functions. There is no such module today, so the rename would be premature.

---

### 2. Single Pipeline Report CSV

**Current state.** Two per-case files must be manually joined to answer any question about a run:

| File | Scope | Join key |
|---|---|---|
| `folder_rename_map.csv` | All cases (including not-converted) | `fixed_name` |
| `ingestion_summary.csv` | All cases (conversion result or synthesised failed) | `exam_name` |

The join key is identical (`fixed_name` == `exam_name`), but the column names differ, so the join is not obvious.

**Proposed `pipeline_report.csv` schema** (19 columns, one row per input case):

```
original_name, fixed_name, data_path, total_dcms,
validation_status, duplicate_slice_count, scout_slice_count,
orientation, modality, slice_thickness, patient_age, patient_sex,
image_dimensions, class, location, exam_size,
conversion_status, conversion_reason, output_path
```

The first 16 columns come from `folder_rename_map.csv` (already populated for every case including non-converted ones via `final_data`). The last three replace the four columns of `ingestion_summary.csv` (`exam_name` is now redundant with `fixed_name`; `status` → `conversion_status`; `reason` → `conversion_reason`).

**How cases that never reached conversion get their `conversion_status`.**
`_build_audit_log` already synthesises rows for these cases. Under the merged design, the mapping is:

| Reason for exclusion | `conversion_status` | `conversion_reason` | `output_path` |
|---|---|---|---|
| Failed DICOM validation (any non-OK `validation_status`) | `"not_attempted"` | the `validation_status` string (e.g. `"VARIABLE_SPACING"`) | `""` |
| Passed validation but no label (`MISSING_CLASS`) | `"not_attempted"` | `"MISSING_CLASS"` | `""` |
| Data code excluded from conversion (`MISSING`, `EMPTY`, `DUPLICATE_DATA`) | `"not_attempted"` | the `data_code` string | `""` |
| Actually converted | `"success"` / `"skipped"` / `"failed"` | as returned by worker | absolute path or `""` |

**Implementation path** (minimal diff):
1. In `data_cleaner.run_pipeline`: replace the separate `write_audit_log(audit_log, ...)` call (step 8) and inline `csv.DictWriter` block (step 9) with a single `logging_utils.write_pipeline_report(audit_log, final_data, path)` call.
2. In `_build_audit_log`: add the three new fields to the synthesised rows using the mapping above.
3. In `logging_utils`: add `write_pipeline_report`; delete `write_audit_log`.
4. `mixed_series_analysis.csv` is per-series (not per-case) and stays as a separate file — it has no natural merge target.

---

### 3. Logging Simplification

**What the `_JsonlHandler` and `log_jsonl` cost vs. what they provide.**

`_JsonlHandler` is 30 lines of code (the `emit` override, the `try/except`, the `self.flush()` call, and the JSON serialisation) that produce `ingestion.log` in JSONL format. No script in the repository reads or processes this file — the JSONL structure provides zero benefit over plain text. `log_jsonl` (lines 193–215 of `logging_utils.py`) is explicitly dead code and mutates its caller's dict in place as a side effect.

**Exact stdlib replacement for `_JsonlHandler`:**

Delete the entire `_JsonlHandler` class and replace the two lines that instantiate it in `setup_logger` with:

```python
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
))
```

This produces a log file that is human-readable, grep-friendly, and contains the same four fields (`timestamp`, `level`, `logger`, `message`) that `_JsonlHandler` emits — but as a single plain-text line instead of a JSON object. The crash-safety property (`self.flush()` after every write) is preserved because `logging.FileHandler` flushes after every `emit()` by default in CPython.

**Full removal list:**

| Symbol | File | Action |
|---|---|---|
| `_JsonlHandler` (class) | `logging_utils.py` | Delete |
| `log_jsonl` (function) | `logging_utils.py` | Delete |
| `write_audit_log` (function) | `logging_utils.py` | Delete once `write_pipeline_report` is added (see §2) |
| `import json` | `logging_utils.py` | Remove (only used by deleted symbols) |
| `from datetime import datetime` | `logging_utils.py` | Remove (only used by deleted symbols) |

After these deletions and the addition of `write_pipeline_report`, `logging_utils.py` contains exactly two functions: `setup_logger` (~15 lines) and `write_pipeline_report` (~20 lines). The module shrinks from ~215 lines to ~45 lines.

**What to keep unchanged:** the `if logger.handlers: return logger` guard in `setup_logger` is justified and should not be removed — it prevents duplicate handlers in test runs.

---

### 4. Diagnostics Consolidation

**The nine scripts and their disposition:**

| Script | Disposition | Reason |
|---|---|---|
| `survey_nifti.py` | Merge into `diagnose.py survey` | Core survey step; all other diagnostics depend on its output. |
| `analyse_nifti_survey.py` | Merge into `diagnose.py analyse` | Reads `nifti_survey.csv`; same dep set as `survey_nifti.py`. |
| `check_nifti.py` | Merge into `diagnose.py check FILE` | 14 lines; no argparse; trivially wraps into a subcommand function. |
| `check_nifti_quality.py` | Merge into `diagnose.py quality FILE` | Shares nibabel/numpy deps; the hardcoded `NIFTI_DIR` constant must become a CLI argument in the merge. |
| `check_resize_quality.py` | Merge into `diagnose.py resize` | Same dep set; reads CSVs produced by `survey`. |
| `check_fov_air.py` | Merge into `diagnose.py fov-air` | Same dep set; depends on `survey` output. |
| `compute_optimal_spacing.py` | Merge into `diagnose.py spacing` | Reads `nifti_survey.csv`; produces a console report, no file output. |
| `extract_gender.py` | **Delete** | Fully superseded by `inventory_scan.write_gender_to_classes_csv`. Keeping it creates two authoritative sources for the same operation with no benefit. |
| `check_missing_gender.py` | **Delete** | 12-line script using `pandas` for a cross-reference that `inventory_scan._log_status_breakdown` already produces automatically at the end of every inventory run. The `pandas` dependency exists nowhere else in the pipeline scripts and is the only reason it cannot be trivially inlined. |

**Specific overlaps that justify the two deletions:**
- `extract_gender.py` superseded: `inventory_scan.write_gender_to_classes_csv` performs the same PatientSex → numeric-gender → `classes.csv` backfill, running inline during every inventory scan. The `inventory_scan.py` docstring already notes this relationship explicitly.
- `check_missing_gender.py` superseded: `inventory_scan._log_status_breakdown` logs the count of `gender_status == "MISSING"` cases at the end of every `run_inventory` / `run_partitioned` call. The 12-line script adds no information that is not already produced automatically.

**`diagnose.py` proposed CLI surface:**

```
python data_engine/diagnostics/diagnose.py survey   [--nifti-dir DIR] [--output-dir DIR]
python data_engine/diagnostics/diagnose.py analyse  [--survey-csv PATH] [--output-dir DIR]
python data_engine/diagnostics/diagnose.py check    FILE
python data_engine/diagnostics/diagnose.py quality  FILE [--nifti-dir DIR]
python data_engine/diagnostics/diagnose.py resize   [--survey-csv PATH] [--dataset-base DIR]
python data_engine/diagnostics/diagnose.py fov-air  [--nifti-dir DIR] [--survey-csv PATH]
python data_engine/diagnostics/diagnose.py spacing  [--survey-csv PATH]
```

Scripts that should **stay separate** rather than being merged: none of the remaining seven have any reason to stay as standalone files — they share a dependency set, a target audience (data analyst running manual checks), and a data source (the `nifti/` directory). The `diagnose.py` entry point is a standard `argparse` subcommand pattern and requires no new dependencies.

---

### 5. `inventory_scan.py` / `data_cleaner.py` Deduplication

**The exact duplication.** `add_class_status` in `inventory_scan.py` (lines 64–118) performs the following steps:
1. Open `classes.csv` with `csv.DictReader`.
2. Build a `set[str]` of uppercased exam names that have a non-empty `class` value.
3. For each case record, regex-strip the disambiguation suffix (`BP001A` → `BP001`) using `re.match(r"(BP\d+)", ...)`.
4. Classify as `"OK"` or `"MISSING"` based on set membership.

Steps 1–3 are already performed by `class_utils.join_class_data`, which builds a `dict` keyed by the same uppercased, regex-stripped exam names and merges `class`, `location`, and `patient_age` into each record that matches.

**Minimal change to eliminate the duplication:**

`add_class_status` should not re-read `classes.csv` or re-strip suffixes. Instead, it should run as a post-join pass that inspects the `class` key already merged by `join_class_data`:

```python
def add_class_status(data: list[dict], logger) -> list[dict]:
    """Post-join pass: set class_status based on whether join_class_data
    attached a class key. Must be called after join_class_data."""
    missing_count = 0
    for item in data:
        if item.get("class"):          # join_class_data already did the lookup
            item["class_status"] = "OK"
        else:
            item["class_status"] = "MISSING"
            missing_count += 1
    logger.info(f"  - {missing_count} case(s) have no class label in classes.csv.")
    return data
```

The `classes_csv` parameter is dropped entirely. The call sites in `run_inventory` and `run_partitioned` both already call `join_class_data` before `add_class_status`, so the sequencing constraint is already satisfied. This removes the duplicate file-open, the duplicate dict-build, and the duplicate regex — 25 lines of logic collapse to 10.

The important behavioural distinction is preserved: `add_class_status` still writes to `class_status` (not `validation_status`), so a case with `validation_status = "VARIABLE_SPACING"` and no label correctly ends up with both `validation_status = "VARIABLE_SPACING"` and `class_status = "MISSING"`.

---

### 6. Long Comments

**Top offenders and what to cut:**

**`nifti_utils.py` — `convert_series_to_nifti` docstring (~60 lines).**
Sections to cut entirely: "What is the affine matrix?" (8 lines of prose explaining what an affine is — domain education, not function documentation) and "Float32 casting:" (5 lines explaining that float32 is half the size of float64 — universally known). What to keep: one sentence each for the four design choices (ITKReader rationale, float32 dtype, affine preservation, explicit `del` + `gc.collect()`). Target: 10–12 lines.

**`nifti_utils.py` — module-level docstring (lines 1–52).**
Cut: the "What is NIfTI and why convert to it?" section (lines 14–20) and "What is the affine matrix?" prose — both belong in a project README, not a module docstring. Keep: the one-sentence module purpose statement and the public API table (the `generate_variant_*` table already present at the bottom). Target: 8 lines.

**`logging_utils.py` — module-level docstring (lines 2–38).**
This entire docstring describes the "Two output formats, two audiences" design and the "Pipeline audit CSV" rationale. Once `_JsonlHandler` and `log_jsonl` are removed (§3) and `write_audit_log` is replaced by `write_pipeline_report` (§2), roughly 30 of these 37 lines become stale. Replace with a three-sentence summary: what the module does, what `setup_logger` returns, and what `write_pipeline_report` writes. Target: 5 lines.

**`logging_utils.py` — `_JsonlHandler` docstring (lines 52–71).**
Moot if the class is deleted (§3). If it is kept, the docstring can be reduced to one sentence: the class is a `FileHandler` subclass that serialises each record as a JSONL line. The two-paragraph explanation of "why not use stdlib directly" adds no information beyond what the class name already communicates.

**`logging_utils.py` — `setup_logger` docstring — duplicate-handler guard explanation (~8 lines).**
Replace the paragraph that explains the `if logger.handlers:` guard with one sentence: "Guards against re-adding handlers when called more than once in the same process (e.g. during testing)."

**`data_cleaner.py` — mount-guard inline comment in `run_pipeline` (7 lines).**
The `RuntimeError` message already provides a user-facing explanation. The 7-line comment explaining why the check exists can be replaced with one line: `# Prevent silent writes to local disk when the external SSD is unmounted.`

**`inventory_scan.py` — `_run_pipeline_steps` docstring — MISSING-row note.**
The note that MISSING sentinel rows are stripped from the return value appears in the docstring, in an inline comment at line 279, and in the "What `_run_pipeline_steps` does" doc section. Pick the docstring as the single canonical location; remove the inline comment and the doc-section repetition.

---

### 7. Library Consolidation

**`typing` imports replaceable with built-in generics.**

`nifti_resize.py` line 68:
```python
# Before (pre-3.9 style)
from typing import Dict, Tuple

# After (Python 3.9+ built-in generics)
# No import needed — use dict and tuple directly in annotations
```

Specifically: `Dict[str, Tuple[int,int,int]]` → `dict[str, tuple[int,int,int]]` and `Tuple[int,int,int]` → `tuple[int, int, int]` throughout the file. The rest of the codebase already uses the built-in generic syntax (e.g. `list[dict]`, `str | Path`), so this is a consistency fix. No other `typing` imports exist in `src/`.

**`pandas` removal.**

`pandas` is used in exactly one file: `diagnostics/check_missing_gender.py`, for `pd.read_csv` and `DataFrame.isin`. The recommended action is to delete this script (§4). If it must be kept for any reason, the replacement with stdlib is trivial:

```python
import csv

with open(classes_csv) as f:
    labelled = {row["exam"].upper() for row in csv.DictReader(f) if row.get("gender")}

with open(survey_csv) as f:
    all_exams = {row["fixed_name"].upper() for row in csv.DictReader(f)}

missing = all_exams - labelled
```

`pandas` has no presence in any pipeline module (`src/`) and no presence in any other diagnostics script, so deleting `check_missing_gender.py` eliminates the `pandas` dependency from the diagnostics directory entirely.

**Centralising the `r"(BP\d+)"` regex.**

The pattern appears in three places with three slightly different forms:

| File | Line | Form |
|---|---|---|
| `data_engine/src/file_utils.py` | 219 | `re.compile(r"BP(\d+)")` (compiled, no outer group) |
| `data_engine/src/class_utils.py` | 110 | `re.match(r"(BP\d+)", ...)` (inline, outer group) |
| `data_engine/inventory_scan.py` | 108, 197 | `re.match(r"(BP\d+)", ...)` (inline, outer group, twice) |

`file_utils.py` is the right home because it defines the `BP{NNN}` naming convention and is already imported by both `class_utils.py` and `inventory_scan.py`. The canonical form should be:

```python
# file_utils.py — module level
_BP_BASE_RE = re.compile(r"BP(\d+)")
```

`class_utils.py` and `inventory_scan.py` then import and use `_BP_BASE_RE` instead of redefining the pattern. The outer-group vs. inner-group discrepancy (the two inline uses capture the full `BP001`, while `file_utils` captures only the digit sequence) should be resolved to one canonical form at consolidation time. The outer-group form (`r"(BP\d+)"`, capturing `BP001`) is more useful at call sites that need the full base name, so prefer that.
