## dicom_utils.py

**Location:** `data_engine/src/dicom_utils.py`

**Purpose:** Validates raw DICOM series for geometric integrity and produces a breakdown report for folders that contain multiple mixed series. Only cases that pass all checks are forwarded to the NIfTI conversion step.

---

### Libraries Used

| Library | Role |
|---|---|
| `pydicom` / `pydicom.errors.InvalidDicomError` | Reading DICOM file headers; catching parse failures |
| `numpy` | Vector math (cross product, dot product, diff, unique, isclose) for projection-based sorting and spacing analysis |
| `os` | Directory listing, path joining |
| `csv` | Writing the mixed-series analysis CSV |
| `collections.defaultdict` | Grouping DICOM files by SeriesInstanceUID in `analyze_mixed_series` |
| `logging` | Structured progress and warning messages via the `dicom_ingestion` logger |

---

### Function Reference

---

#### `get_orientation(iop)`

**What it does:** Classifies a DICOM scan plane (AXIAL, CORONAL, SAGITTAL, OBLIQUE, or UNKNOWN) from the six-float `ImageOrientationPatient` tag.

**Design rationale:** Real scanner output is never perfectly aligned; a patient's slight tilt produces values such as `[0.9998, 0.0017, 0, ...]` instead of the ideal `[1, 0, 0, ...]`. Rounding each component to the nearest integer before matching absorbs these physical imperfections without misclassifying genuinely oblique acquisitions.

**Parameters:**
- `iop` — `list[float] | None` — The `ImageOrientationPatient` DICOM tag value (6 floats), or `None` if the tag is absent.

**Returns:** `str` — one of `'AXIAL'`, `'CORONAL'`, `'SAGITTAL'`, `'OBLIQUE'`, `'UNKNOWN'`.

**Side effects / exceptions:** None. Returns `'UNKNOWN'` defensively for `None` or short lists.

**Complexity flag:** The `if/elif` chain over a fixed set of signatures should be replaced with a module-level dict lookup, eliminating the chain and making future orientation additions trivial:
```python
_IOP_MAP = {
    (1, 0, 0, 0, 1, 0):  'AXIAL',
    (1, 0, 0, 0, 0, -1): 'CORONAL',
    (0, 1, 0, 0, 0, -1): 'SAGITTAL',
}
return _IOP_MAP.get(tuple(round(x) for x in iop), 'OBLIQUE')
```

---

#### `load_dicom_metadata(path)`

**What it does:** Reads the DICOM header (not the pixel array) of every `.dcm` file in a directory and returns the successfully parsed headers as a list of `pydicom.Dataset` objects.

**Design rationale:** `stop_before_pixels=True` skips the large pixel data array, making each file load roughly 10-100x faster. Corrupt files are silently skipped rather than aborting because a single bad file in a folder of hundreds should not invalidate the entire series.

**Parameters:**
- `path` — `str` — Absolute path to the directory containing `.dcm` files.

**Returns:** `list[pydicom.Dataset]` — one entry per successfully parsed file; files that raise `InvalidDicomError` are omitted.

**Side effects / exceptions:** Logs a `WARNING` for each unparseable file. Does not raise.

**Flags:**
- `os.listdir` + manual `.endswith('.dcm')` filtering could be replaced with `pathlib.Path.glob('*.dcm')`.
- This function and `validate_dcms` both call `os.listdir` on the same path, listing the directory twice.

---

#### `_sort_slices_by_projection(metadata_list, iop)` _(private)_

**What it does:** Sorts a list of DICOM slice headers into correct anatomical order by projecting each slice's real-world 3D position onto the scan's stacking axis.

**Design rationale:** `InstanceNumber` is frequently reset or scrambled by PACS systems during export and cannot be trusted. `ImagePositionPatient` records the actual 3D patient-coordinate origin of each slice in millimetres. The stacking axis is derived from `ImageOrientationPatient` via a cross product of the row and column direction vectors; projecting each slice's position onto this axis yields a physically grounded, scanner-independent sort key.

**Parameters:**
- `metadata_list` — `list[pydicom.Dataset]` — Headers for a single series. Modified in-place: sorted and `.dist` (float, mm) added to each entry.
- `iop` — `list[float]` — `ImageOrientationPatient` tag value (6 floats).

**Returns:** `None`.

**Side effects / exceptions:** Mutates `metadata_list` (in-place sort) and adds `.dist` to every `Dataset`. Sets `.dist = 0` as a failsafe for any slice missing `ImagePositionPatient`.

---

#### `_evaluate_spacing(metadata_list)` _(private)_

**What it does:** Computes inter-slice distances from the `.dist` attributes and classifies the series geometric integrity into one of five statuses: `OK`, `GAPPED_SEQUENCE`, `GAPPED_SEQUENCE (N missing)`, `DUPLICATE_SLICES`, or `VARIABLE_SPACING`.

**Design rationale:** The DICOM `SliceThickness` tag diverges from actual reconstructed spacing when overlapping reconstructions are used, so spacing is computed empirically from real-world coordinates. Rounding deltas to 2 decimal places absorbs floating-point noise. The `GAPPED_SEQUENCE (N missing)` branch detects exactly one missing slice between some pairs (2x spacing), counting the gaps for reviewer context.

**Parameters:**
- `metadata_list` — `list[pydicom.Dataset]` — Sorted slice headers with `.dist` attributes populated.

**Returns:** `tuple[str, np.ndarray, int, np.ndarray]`
- `validation_status` — string classification
- `unique_spacings` — `np.ndarray` of unique rounded inter-slice distances (mm)
- `duplicate_count` — `int`, number of zero-distance slice pairs
- `deltas` — `np.ndarray` of all raw absolute inter-slice distances (mm)

**Side effects / exceptions:** None. Pure computation.

**Complexity note:** The `if/elif/else` chain over `len(unique_spacings)` is readable and maps directly to the classification logic. No significant simplification warranted, though early-return style would make the `VARIABLE_SPACING` fallthrough more explicit.

---

#### `validate_dcms(data_mapping, base_path)`

**What it does:** Runs the full ten-step DICOM series validation pipeline on every case record in `data_mapping`, updating each record in-place with a `validation_status` and extracted metadata, then applies a physical exam-size outlier check.

**Design rationale:** Checks are ordered from cheapest to most expensive and from most catastrophic to most nuanced; early `continue` statements short-circuit remaining checks on the first disqualifying condition. The `NOT_APPLICABLE` guard allows this function to be called on the full unfiltered case list. `effective_slice_thickness` is computed from the modal inter-slice delta rather than the `SliceThickness` tag for the same reason as in `_evaluate_spacing`. Outlier detection is deferred to after all cases are processed so it operates on the complete OK-case distribution.

**Parameters:**
- `data_mapping` — `list[dict]` — Case record dicts as produced by `file_utils.organize_data`. Each dict is updated in-place.
- `base_path` — `str` — Absolute path to the raw DICOM root directory.

**Returns:** `list[dict]` — The same list reference with validation fields merged into every record.

**Side effects / exceptions:** Mutates every dict in `data_mapping`. Logs progress at `INFO` and warnings for scout-filtered and single-slice cases. The outer `try/except Exception` catches all unhandled per-case errors and records them as `VALIDATION_ERROR: <message>` rather than aborting the pipeline.

**Flags:**
- `os.listdir` is called to check for `.dcm` files, then `load_dicom_metadata` calls it again — consolidating to one call would be cleaner.
- `isinstance(effective_slice_thickness, float)` type guard is unusual; a `None` sentinel would be more idiomatic.

---

#### `_flag_exam_size_outliers(data_mapping)` _(private)_

**What it does:** Reclassifies `OK`-status cases as `BELOW_LIMIT` or `ABOVE_LIMIT` if their physical exam coverage falls outside hardcoded anatomical bounds (90 mm minimum, 300 mm maximum).

**Design rationale:** The bounds are fixed anatomical constants rather than IQR-derived statistical fences. An earlier IQR-based approach was introduced to catch one pathological case (a 4000-slice exam from mixed series) but had the side effect of flagging valid exams whenever the dataset distribution shifted. The absolute limits reflect real anatomy: scans shorter than 90 mm are partial acquisitions or scouts; scans longer than 300 mm after scout filtering are almost certainly mis-exports.

**Parameters:**
- `data_mapping` — `list[dict]` — Full list of case dicts from `validate_dcms`.

**Returns:** `list[dict]` — The same list, with `validation_status` updated in-place for outlier cases.

**Side effects / exceptions:** Mutates `validation_status` of outlier dicts. Logs outlier cases at `INFO` level. No exceptions raised.

**Flags:**
- The `exam_sizes` list is populated but only used for an early-return guard. The guard's intent would be clearer as `if not ok_exams`.
- `isinstance(item.get('exam_size'), (int, float, np.number))` includes `np.number` to handle numpy scalar types not explicitly cast to Python floats upstream — legitimate defensive measure.

---

#### `analyze_mixed_series(folder_path)`

**What it does:** Groups all DICOM files in a mixed-series folder by `SeriesInstanceUID` and runs a mini-validation on each group, returning a list of per-series report dicts.

**Design rationale:** When `validate_dcms` finds multiple UIDs it cannot choose the correct diagnostic series automatically — human judgment is required. This function produces the information needed for that decision. It deliberately reuses `_sort_slices_by_projection` and `_evaluate_spacing` rather than duplicating logic.

**Parameters:**
- `folder_path` — `str` — Absolute path to the mixed-series case folder.

**Returns:** `list[dict]` — One report dict per `SeriesInstanceUID`. Returns `[]` if the path does not exist or no DICOM files can be read.

**Side effects / exceptions:** Catches all per-series exceptions as `VALIDATION_ERROR: <message>`. No file writes.

---

#### `analyze_mixed_folders(data_mapping, base_path, output_csv_path)`

**What it does:** Collects all cases flagged as `MIXED_SERIES_ERROR`, calls `analyze_mixed_series` on each, and writes the combined per-series report to a CSV file for manual review.

**Design rationale:** The CSV is the handoff between the automated pipeline and human review. Column names are collected dynamically across all report dicts rather than being hardcoded, making the CSV robust to series that are missing optional fields.

**Parameters:**
- `data_mapping` — `list[dict]` — Full case list from `validate_dcms`.
- `base_path` — `str` — Absolute path to the raw DICOM root directory.
- `output_csv_path` — `str` — Absolute path where the mixed-series CSV will be written.

**Returns:** `None`.

**Side effects / exceptions:** Creates or overwrites the file at `output_csv_path`. `open()` can raise `OSError` if the path is not writable; this is not caught.

---

### Cross-Cutting Notes

**Simplification opportunities:**

1. `get_orientation` — `if/elif` chain over a closed set → replace with a module-level dict lookup.
2. `validate_dcms` and `load_dicom_metadata` — `os.listdir` called twice on the same directory.
3. `validate_dcms` — `isinstance(effective_slice_thickness, float)` → use `None` sentinel.
4. `_flag_exam_size_outliers` — `exam_sizes` early-return guard → rename to `ok_exams` to express intent directly.

**No function raises intentional exceptions.** All error paths are handled by returning early with a sentinel value or recording a status string in the output dict — a deliberate pipeline-safety design so a single malformed case does not abort validation of hundreds of others.
