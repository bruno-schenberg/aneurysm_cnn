## nifti_utils.py

**Location:** `data_engine/src/nifti_utils.py`

**Module purpose:** Converts validated DICOM series into compressed NIfTI volumes (`.nii.gz`) and writes them into the class-labelled directory layout expected by the training engine. Conversion is parallelised across subprocesses via `ProcessPoolExecutor`.

---

### Function inventory

#### `filter_for_conversion(exam_data)`

**What it does:** Filters the full list of case record dicts down to only those that are safe and labelled enough to convert.

**Design rationale:** Two independent gates are enforced in a single list comprehension. The validation gate (`validation_status == 'OK'`) prevents geometrically corrupt DICOM series from producing bad volumes. The label gate (`class in ('0', '1')`) prevents unlabelled cases from silently entering the training directory, which would corrupt dataset integrity without raising an error elsewhere.

**Parameters:**
- `exam_data` — `list[dict]`: Full list of case records after both validation and class-joining have been completed. Each dict is expected to carry at minimum the keys `'validation_status'` and `'class'`.

**Returns:** `list[dict]` — Subset of `exam_data` where both conditions are met. May be empty.

**Side effects:** Logs one INFO line with the count of eligible exams.

**Exceptions:** None raised.

**Category:** Pre-conversion filtering (not I/O, not quality checking).

---

#### `convert_series_to_nifti(dicom_dir, output_path)`

**What it does:** Loads a DICOM series directory with MONAI + ITKReader, casts the volume to float32, attaches the reconstructed affine matrix, and saves a compressed NIfTI file.

**Design rationale:**
- **ITKReader** is used instead of raw pydicom because ITK has two decades of production use for DICOM series reconstruction. It handles slice ordering, multi-frame/enhanced DICOM edge cases, and affine matrix construction from `ImagePositionPatient`, `ImageOrientationPatient`, and `PixelSpacing` tags automatically.
- **Float32 casting** standardises the dtype across scanners and matches the dtype PyTorch and MONAI training transforms expect, avoiding an extra cast at training time.
- **Affine preservation** ensures every downstream tool knows the physical size, orientation, and origin of each voxel.
- **Explicit `del` + `gc.collect()`** after saving: ITK and MONAI hold large allocations on the C++ heap that Python's GC cannot see. Without this, memory accumulates when many cases are processed sequentially inside a single subprocess.
- **`OSError` is re-raised** so the caller (`_convert_exam_worker`) can distinguish a disk/permission failure from a DICOM parsing failure.

**Parameters:**
- `dicom_dir` — `str`: Absolute path to the directory containing `.dcm` files for one series.
- `output_path` — `str`: Absolute path where the `.nii.gz` file will be written.

**Returns:** `bool` — `True` on success, `False` if a non-IO exception occurred during loading or conversion.

**Side effects:**
- Writes a `.nii.gz` file to `output_path`.
- Logs an INFO line on success, an ERROR line on failure.
- Forces a garbage-collection cycle on success.

**Exceptions:**
- `OSError` — re-raised on disk-full, permission-denied, or other filesystem errors.
- All other exceptions — caught, logged, indicated via `False` return value.

**Category:** NIfTI I/O (primary). No validation or quality-checking logic — trusts that filtering has already occurred upstream.

---

#### `_convert_exam_worker(exam, dicom_base_path, nifti_output_dir)`

**What it does:** Converts a single exam from DICOM to NIfTI; designed to run inside a subprocess worker.

**Design rationale:**
- **Subprocess isolation:** Cannot safely write to the main process logger. Log messages are collected in a list inside the result dict and flushed by the main process after the future completes.
- **Idempotency:** An existence check on the output path means the pipeline can be interrupted and re-run safely; already-converted files are left untouched and the case is marked `'skipped'`.
- **Separate `OSError` vs. general exception handling** produces distinct `'reason'` strings (`'FAILED_IO'` vs. a descriptive message), making it easier to triage failures.
- **Output path convention:** `{nifti_output_dir}/{class}/{fixed_name}.nii.gz` — the class subdirectory is created with `exist_ok=True` inside the worker so workers are self-contained.

**Parameters:**
- `exam` — `dict`: Case record with at minimum the keys `'fixed_name'`, `'class'`, and `'data_path'`.
- `dicom_base_path` — `str`: Absolute path to the DICOM root directory.
- `nifti_output_dir` — `str`: Absolute path to the NIfTI output root directory.

**Returns:** `dict` with keys:
- `'exam_name'` (`str`) — canonical case name
- `'status'` (`str`) — `'success'`, `'skipped'`, or `'failed'`
- `'reason'` (`str`) — empty on success; error description on failure
- `'output_path'` (`str`) — absolute path to the output file; empty string on failure
- `'log_messages'` (`list[str]`) — log lines to be flushed by the main process; **removed before results are returned to callers**

**Side effects:**
- Creates `{nifti_output_dir}/{class}/` directory if absent.
- Delegates to `convert_series_to_nifti`, which writes a `.nii.gz` file.

**Exceptions:** All exceptions are caught internally and encoded in `'status'`/`'reason'`; nothing propagates out of the worker.

**Category:** NIfTI I/O orchestration (subprocess unit). Mixes path construction, idempotency check, and I/O delegation — see flags below.

---

#### `process_and_convert_exams(eligible_exams, dicom_base_path, nifti_output_dir, max_workers=None)`

**What it does:** Dispatches all eligible exams to a `ProcessPoolExecutor`, collects results as each future completes, and returns a cleaned result list.

**Design rationale:**
- **`ProcessPoolExecutor` over `ThreadPoolExecutor`:** MONAI and ITK manage their own internal C++ thread pools; running them from Python threads causes contention. Separate processes have fully independent memory spaces and thread pools.
- **`as_completed` over ordered iteration:** Results are collected and logged as soon as each exam finishes, giving real-time progress feedback during long runs.
- **Log-message flushing:** `result.pop("log_messages", [])` flushes each worker's deferred log lines into the main logger and strips that key before appending to the output, keeping the public result schema clean.
- **`max_workers=None`** defaults to `os.cpu_count()`. Should be reduced if RAM is constrained, since each worker holds one full 3D volume in memory simultaneously.

**Parameters:**
- `eligible_exams` — `list[dict]`: Case records as returned by `filter_for_conversion`.
- `dicom_base_path` — `str`: Absolute path to the DICOM root directory.
- `nifti_output_dir` — `str`: Absolute path to the NIfTI output root directory; created automatically if absent.
- `max_workers` — `int | None`: Number of parallel worker processes; `None` uses all available CPUs.

**Returns:** `list[dict]` — one result dict per exam with keys `'exam_name'`, `'status'`, `'reason'`, `'output_path'`.

**Side effects:**
- Creates `nifti_output_dir` if it does not exist.
- Spawns up to `max_workers` subprocesses, each of which may create subdirectories and write `.nii.gz` files.
- Logs progress and a final summary.

**Category:** NIfTI I/O orchestration (parallel driver).

---

### Categorisation by concern

| Function | Category |
|---|---|
| `filter_for_conversion` | Pre-conversion filtering |
| `convert_series_to_nifti` | NIfTI I/O |
| `_convert_exam_worker` | NIfTI I/O orchestration (subprocess unit) |
| `process_and_convert_exams` | NIfTI I/O orchestration (parallel driver) |

There are **no pure quality-checking or metadata-inspection functions** in this module. Quality checking is fully delegated to upstream (`dicom_utils.validate_dcms`); `filter_for_conversion` merely reads the result.

---

### Flags: mixed responsibilities

**`_convert_exam_worker`** mixes three concerns in one function:
1. Output path construction and directory creation.
2. Idempotency check (existence test on the output file).
3. Delegation to `convert_series_to_nifti` for actual I/O.

This is a reasonable trade-off for a private subprocess worker, but if the module grows, points 1 and 2 should be extracted into helpers to keep the worker focused on orchestration.

---

### Flags: dead code

The `futures` dict in `process_and_convert_exams` maps future objects to exam names but is never read — it would only be useful inside a `try/except` around `future.result()`, which is absent. Could be removed without changing behaviour.
