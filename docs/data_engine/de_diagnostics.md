## diagnostics/check_fov_air.py

### Problem being diagnosed

Some CT exams nominally report a large FOV (e.g. 240 mm) but a significant portion of that volume is empty air on one or more faces. This script measures how deep the air border is on each of the six faces of every NIfTI volume and derives the true anatomical content width per axis (`fov_mm - air_low_mm - air_high_mm`). The results let downstream steps detect exams where the usable brain extent is materially smaller than the declared FOV, which would otherwise cause crop/resize artefacts.

---

### Inputs

| File | CLI flag | Purpose |
|---|---|---|
| `nifti_survey.csv` | `--survey-csv` | Per-exam metadata: `dim_x/y/z`, `spacing_x/y/z_mm`, `class`. One row per exam. |
| `nifti_survey_groups.csv` | `--groups-csv` | Scanner-group assignments produced by `survey_nifti.py`: `group_id`, `group_count`, representative `dim_x/y`, `spacing_x/z_mm`. |
| `*.nii.gz` files | `--nifti-dir` | The actual NIfTI volumes, located under `<nifti-dir>/<exam>/<exam>.nii.gz` (recursive glob `*/*.nii.gz`). |

### Output

| File | CLI flag | Contents |
|---|---|---|
| `fov_air.csv` | `--output` | One row per successfully processed exam with 28 columns: identity fields (`exam`, `class`, `group_id`, `group_count`), dimensions, spacings, FOV in mm, air depth in voxels and mm for each of the 6 faces, and derived content width in mm for each axis. |

A per-group summary table is also printed to stdout (median air depths and content widths).

---

### Functions

#### `air_depth(data, axis, from_high, threshold) -> int`

Counts how many consecutive all-air slices lie against one face of the volume along a given axis.

| Parameter | Type | Description |
|---|---|---|
| `data` | `np.ndarray` (3-D, float32) | HU voxel array. |
| `axis` | `int` | Axis to scan along: 0 = X, 1 = Y, 2 = Z. |
| `from_high` | `bool` | `True` to scan inward from the high-index face; `False` from index 0. |
| `threshold` | `float` | A slice is air when its maximum HU value is `<= threshold` (default -200 HU). |

**Returns:** `int` — number of air slices from that face (0 if the outermost slice already contains signal; `n` if the entire volume is air).

**Side effects:** None.

---

#### `assign_group(exam_row, group_rows) -> dict | None`

Finds the scanner group whose representative dimensions and spacing match those of a single exam, using the same tolerances as `survey_nifti.py`.

| Parameter | Type | Description |
|---|---|---|
| `exam_row` | `dict` | One row from `nifti_survey.csv` (keys: `dim_x`, `dim_y`, `spacing_x_mm`, `spacing_z_mm`). |
| `group_rows` | `list[dict]` | All rows loaded from `nifti_survey_groups.csv`. |

Matching criteria (all three must hold):
- `dim_x` and `dim_y` are identical integers.
- `spacing_x_mm` within 15% relative difference of the group representative.
- `spacing_z_mm` within 20% relative difference of the group representative.

**Returns:** `dict` — the first matching group row, or `None` if no group matches.

**Side effects:** None.

---

#### `process_exam(nifti_path, survey_row, group_rows, threshold) -> dict`

Loads one NIfTI file, measures air depth on all six faces, and assembles a result row.

| Parameter | Type | Description |
|---|---|---|
| `nifti_path` | `pathlib.Path` | Absolute path to the `.nii.gz` file. |
| `survey_row` | `dict` | Corresponding row from `nifti_survey.csv`. |
| `group_rows` | `list[dict]` | All rows from `nifti_survey_groups.csv` (passed through to `assign_group`). |
| `threshold` | `float` | HU threshold forwarded to `air_depth`. |

**Returns:** `dict` with all 28 keys listed in `FIELDNAMES`.

**Side effects:**
- Loads the full NIfTI volume into memory as float32 (`mmap=False`).
- Explicitly deletes the array and calls `gc.collect()` before returning to limit peak memory usage.

---

#### `main(nifti_dir, survey_csv, groups_csv, output_csv, threshold) -> None`

Orchestrates the full pipeline: loads CSVs, iterates over all NIfTI files, writes `fov_air.csv`, and prints a per-group summary table.

| Parameter | Type | Description |
|---|---|---|
| `nifti_dir` | `str` | Root directory searched recursively for `*/*.nii.gz`. |
| `survey_csv` | `str` | Path to `nifti_survey.csv`. |
| `groups_csv` | `str` | Path to `nifti_survey_groups.csv`. |
| `output_csv` | `str` | Destination path for `fov_air.csv`. |
| `threshold` | `float` | HU threshold passed to `process_exam`. |

**Returns:** None.

**Side effects:**
- Reads `survey_csv` and `groups_csv` into memory at startup (file handles are not explicitly closed — relies on garbage collection).
- Prints per-exam progress to stdout using `\r` overwrite.
- Prints warnings to stdout and errors to stderr for missing/failed exams.
- Writes `fov_air.csv` (overwrites if it exists).
- Prints a formatted per-group summary table to stdout.
- Calls `sys.exit(1)` if no `.nii.gz` files are found.
- Imports `collections.defaultdict` inside the function body rather than at the top of the module.

---

### Complexity flags

1. **`assign_group` linear scan.** The function iterates over every row in `group_rows` for each exam. With a large number of groups this is O(exams × groups). A dictionary keyed by `(dim_x, dim_y)` would reduce the inner loop to only groups sharing the same matrix size, which is typically a small set.

2. **`air_depth` called six times per exam with full array in memory.** Each call uses `np.take`, which creates a view/copy per slice inside the loop. For very large volumes this is the dominant cost. The six calls could be batched or the array pre-transposed to avoid repeated axis switching, though the current code is readable and correct.

3. **Unclosed file handles in `main`.** `open(survey_csv)` and `open(groups_csv)` are passed directly to `csv.DictReader` without a `with` block. The files are eventually closed by the garbage collector, but using context managers would be cleaner and safer.

4. **`med` lambda defined inside a `for` loop.** The helper `def med(key): ...` is redefined on every iteration of the group-summary loop. Moving it outside the loop (or making it a module-level helper) avoids repeated function object creation and makes the intent clearer.

5. **Late `import` of `collections.defaultdict`.** It is imported inside `main` rather than at the top of the module. This is a minor style issue but can obscure dependencies and confuse linters.

---

## diagnostics/survey_nifti.py

### Problem being diagnosed

This is the **root survey script** for the data engine. It walks every `.nii.gz` file under the raw NIfTI tree (organised as `<nifti-dir>/<class>/<exam>.nii.gz`), reads header metadata without loading voxel data, and writes three CSV files that all downstream diagnostic scripts depend on. The three outputs together answer: _what is in the dataset_ (`nifti_survey.csv`), _what distinct acquisition configurations exist_ (`nifti_survey_variants.csv`), and _how do those configurations cluster into scanner-protocol groups_ (`nifti_survey_groups.csv`).

---

### Inputs

| Source | CLI flag | Default | Purpose |
|---|---|---|---|
| NIfTI directory | `--nifti-dir` | `/mnt/data/nifti` | Root directory; must contain subdirectories `0/` and `1/` holding `.nii.gz` files. |
| Output path | `--output` | `<script_dir>/outputs/nifti_survey.csv` | Destination for the primary CSV; the other two CSVs are derived from this path. |

---

### Output CSVs

#### `nifti_survey.csv`

One row per successfully loaded exam. This is the primary per-exam metadata table consumed by most downstream scripts.

| Column | Type | Description |
|---|---|---|
| `exam` | str | Exam ID stripped of `.nii.gz` extension (double `splitext` to handle `.nii.gz`). |
| `class` | str | Binary label: `"0"` (control) or `"1"` (aneurysm). |
| `dim_x` | int | Voxel count along X axis. |
| `dim_y` | int | Voxel count along Y axis. |
| `dim_z` | int | Voxel count along Z axis. |
| `spacing_x_mm` | float (4 dp) | Voxel spacing along X in mm (from NIfTI header zooms). |
| `spacing_y_mm` | float (4 dp) | Voxel spacing along Y in mm. |
| `spacing_z_mm` | float (4 dp) | Voxel spacing along Z in mm. |
| `size_x_mm` | float (1 dp) | Physical FOV along X = `dim_x * spacing_x_mm`. |
| `size_y_mm` | float (1 dp) | Physical FOV along Y. |
| `size_z_mm` | float (1 dp) | Physical FOV along Z. |
| `min_spacing_mm` | float (4 dp) | `min(spacing_x, spacing_y, spacing_z)`. |
| `max_spacing_mm` | float (4 dp) | `max(spacing_x, spacing_y, spacing_z)`. |
| `is_isotropic` | bool | `True` when `(max_spacing - min_spacing) / max_spacing < 0.05` (within 5%). |

**Downstream consumers:** `analyse_nifti_survey.py`, `check_fov_air.py` (`--survey-csv`), `check_resize_quality.py` (`--survey-csv`), `check_missing_gender.py`.

---

#### `nifti_survey_variants.csv`

One row per unique `(dim_x, dim_y, dim_z, spacing_x, spacing_y, spacing_z)` combination found in the dataset. Produced in the same `main()` call as `nifti_survey.csv`; its path is derived by appending `_variants` before the `.csv` extension.

| Column | Type | Description |
|---|---|---|
| `dim_x` | int | Voxel count along X for this configuration. |
| `dim_y` | int | Voxel count along Y. |
| `dim_z` | int | Voxel count along Z. |
| `spacing_x_mm` | float | Voxel spacing along X in mm. |
| `spacing_y_mm` | float | Voxel spacing along Y in mm. |
| `spacing_z_mm` | float | Voxel spacing along Z in mm. |
| `count` | int | Number of exams that have exactly this configuration. |
| `example` | str | Exam ID of the first exam encountered with this configuration. |

**Downstream consumers:** `check_resize_quality.py` (`--variants-csv`), which picks one representative exam per configuration to test resize quality without exhausting the full dataset.

---

#### `nifti_survey_groups.csv`

One row per unique configuration (same rows as `nifti_survey_variants.csv`) augmented with a scanner-protocol group assignment. Path derived as `_groups.csv`. Produced in the same `main()` call.

| Column | Type | Description |
|---|---|---|
| `group_id` | int | Integer group label assigned by the greedy clustering loop (starts at 1). |
| `group_count` | int | Total number of exams in the group (sum across all configs in the group). |
| `dim_x` | int | Voxel count along X for this configuration. |
| `dim_y` | int | Voxel count along Y. |
| `dim_z` | int | Voxel count along Z. |
| `spacing_x_mm` | float | Voxel spacing along X in mm. |
| `spacing_y_mm` | float | Voxel spacing along Y in mm. |
| `spacing_z_mm` | float | Voxel spacing along Z in mm. |
| `count` | int | Number of exams with exactly this configuration. |
| `example` | str | Representative exam ID for this configuration. |

**Downstream consumers:** `check_fov_air.py` (`--groups-csv`), which re-applies the same grouping tolerances to assign each exam a `group_id` and `group_count` in its output.

---

### Functions

#### `survey_file(nifti_path, label) -> dict`

Loads one NIfTI header and computes per-exam geometry statistics, returning a single row dict ready to be written to `nifti_survey.csv`.

| Parameter | Type | Description |
|---|---|---|
| `nifti_path` | `str` | Absolute path to the `.nii.gz` file. |
| `label` | `str` | Class label (`"0"` or `"1"`), passed through unchanged. |

**Returns:** `dict` with all 14 keys defined in the module-level `FIELDNAMES` list.

**Side effects:** Calls `nib.load()`, which memory-maps the file header but does not load voxel data into RAM. No files are written.

**Design rationale:** Kept as a pure function (no global state, no I/O side effects) so it can be called inside a `try/except` in `main()` without corrupting the output list on failure. The double `os.path.splitext` on the basename (`splitext(splitext(basename)[0])[0]`) strips both `.gz` and `.nii` in one expression, handling the compound extension correctly.

---

#### `main(nifti_dir, output_csv) -> None`

Orchestrates the full survey: iterates over classes `"0"` and `"1"`, calls `survey_file()` for each exam, writes all three CSVs, and prints summary statistics and group tables to stdout.

| Parameter | Type | Description |
|---|---|---|
| `nifti_dir` | `str` | Root directory; expects `<nifti_dir>/0/` and `<nifti_dir>/1/` subdirectories. |
| `output_csv` | `str` | Destination path for `nifti_survey.csv`. The other two CSV paths are derived from this value by string manipulation (`os.path.splitext`). |

**Returns:** None.

**Side effects:**
- Writes `nifti_survey.csv` (overwrites if it exists).
- Writes `nifti_survey_variants.csv` (path derived from `output_csv`).
- Writes `nifti_survey_groups.csv` (path derived from `output_csv`).
- Prints per-axis summary statistics, variant table, and group table to stdout.
- Prints per-file errors to stderr.
- Imports `statistics` inside the function body (lazy import, only executed if there are rows).

**Design rationale:** All three CSV outputs are produced in a single pass over `rows` so the script only needs to load headers once. The variants and groups CSVs are derived path siblings of the primary CSV so the caller only specifies one `--output` path; this keeps the three files co-located and consistently named without requiring additional CLI flags.

---

### Complexity flags

1. **Greedy O(configs²) group assignment loop.** The inner grouping loop (lines 181-195) iterates over every already-assigned config for each new config to find a matching group. This is fine for the tens of unique configurations typical of this dataset but would degrade noticeably for hundreds. A `dict` keyed by `(dim_x, dim_y)` — the first exact-match criterion — would reduce the search to only configs with an identical XY matrix, which is usually a set of one to three entries.

2. **Group summary recomputed via a second linear scan over `group_id`.** The stdout group summary (lines 226-236) re-filters `group_id.items()` with `if g == gid` for every `gid` in the outer loop, making it O(groups × configs). Grouping configs by `gid` into a `dict[int, list]` once before the loop would be cleaner and avoids the repeated scan.

3. **Late import of `statistics`.** `import statistics` appears inside `main()` under a conditional (`if rows:`). Moving it to the top of the module is conventional and avoids hiding the dependency.

4. **Variant CSV sort is on the `seen` dict key, not on `count`.** The `sorted(seen.items())` call on line 158 sorts by the six-tuple `(dim_x, dim_y, dim_z, spacing_x, spacing_y, spacing_z)`, which gives a geometrically ordered table. This is intentional for human readability but could surprise a reader expecting the most-common configurations first.

---

### Hardcoded paths and constants

| Location | Value | Nature |
|---|---|---|
| `argparse` default, line 242 | `--nifti-dir` default = `"/mnt/data/nifti"` | Hardcoded absolute path to the raw data mount. Must be overridden in any environment that differs from the original cluster mount. |
| `argparse` default, line 244-246 | `--output` default = `<script_dir>/outputs/nifti_survey.csv` | Relative to the script's own directory via `os.path.dirname(__file__)`. Robust to being called from any working directory, but the `outputs/` subdirectory must already exist. |
| `survey_file`, line 59 | Isotropic threshold `0.05` (5%) | Magic number embedded in code. Determines whether a volume is classified as isotropic. No CLI flag or named constant exposes it. |
| `main`, line 187 | XY spacing tolerance `0.15` (15%) | Grouping criterion for `spacing_x`; not configurable via CLI. |
| `main`, line 188 | Z spacing tolerance `0.20` (20%) | Grouping criterion for `spacing_z`; not configurable via CLI. |
| `main`, line 83 | Class labels tuple `("0", "1")` | The binary class structure is assumed; a multi-class dataset would require code changes. |

## diagnostics/check_resize_quality.py

### Problem being diagnosed

For each processed dataset variant, this script checks two complementary geometric quality problems:

- **Padding** — the output grid extends beyond the original physical FOV, so a fraction of output voxels must be zero-padded. Variants that add excessive padding waste capacity and dilute the signal.
- **Cropping** — the output grid is smaller than the original physical FOV, so a fraction of the true anatomical extent is discarded. Variants that crop aggressively lose clinical content.

Both metrics are computed geometrically (from header spacings and shapes) rather than by counting zero voxels, because background air in the source scan would otherwise be misclassified as pipeline padding.

---

### Inputs

| File | CLI flag | Purpose |
|---|---|---|
| `nifti_survey_variants.csv` | `--variants-csv` | Representative exam list produced by `survey_nifti.py`; the `example` column identifies one exam per scanner group. |
| `nifti_survey.csv` | `--survey-csv` | Per-exam metadata: `dim_x/y/z`, `spacing_x/y/z_mm`, `size_x/y/z_mm`, `class`. Used to look up the original FOV for each exam. |
| `*.nii.gz` files (raw) | `--raw-nifti-dir` | Raw NIfTI volumes at `<raw-nifti-dir>/<class>/<exam>.nii.gz`. Only exams whose raw file exists here are processed, so the script works against both the full dataset and a small sample. |
| `*.nii.gz` files (processed) | `--datasets-dir` | Root under which all processed dataset folders live (`<datasets-dir>/<ds_folder>/<class>/<exam>.nii.gz`). Missing variant files are silently recorded as empty. |

Datasets checked: raw NIfTI + 16 processed variants — A/B/C/D at 128 voxels; A–F at 192 voxels; A–F at 176 voxels (see `DATASETS` constant).

---

### Output

| File | CLI flag | Contents |
|---|---|---|
| `resize_quality.csv` | `--output` | Long-format CSV: one row per `(dataset_variant, exam)`. Columns: `variant`, `size`, original dimensions/spacings/FOV, output `shape`, per-axis output spacings (`sp_x/y/z`), `pad_pct`, `crop_pct`. Rows for missing variant files have empty metric columns. |

A formatted per-exam and per-dataset aggregate summary (median and max pad/crop %) is also printed to stdout.

---

### Functions

#### `crop_pct(orig_fov, out_shape, out_spacing) -> float`

Computes the fraction of the original physical FOV that was discarded by the resize pipeline.

| Parameter | Type | Description |
|---|---|---|
| `orig_fov` | `tuple[float, float, float]` | Original field-of-view in mm, one value per axis. |
| `out_shape` | `tuple[int, int, int]` | Voxel dimensions of the output NIfTI. |
| `out_spacing` | `tuple[float, float, float]` | Voxel spacing in mm of the output NIfTI. |

**Returns:** `float` in `[0.0, 100.0]` — percentage of original volume cropped away (0 = nothing cropped). The per-axis kept-fractions are multiplied before complementing, giving a volumetric crop estimate.

**Side effects:** None.

---

#### `pad_pct(orig_fov, out_shape, out_spacing) -> float`

Computes the fraction of the output grid that is zero-padding (output extends beyond the original physical FOV).

| Parameter | Type | Description |
|---|---|---|
| `orig_fov` | `tuple[float, float, float]` | Original field-of-view in mm, one value per axis. |
| `out_shape` | `tuple[int, int, int]` | Voxel dimensions of the output NIfTI. |
| `out_spacing` | `tuple[float, float, float]` | Voxel spacing in mm of the output NIfTI. |

**Returns:** `float` in `[0.0, 100.0]` — percentage of output voxels that are geometric padding (0 = no padding added). Per-axis fractions are combined volumetrically, mirroring `crop_pct`.

**Side effects:** None.

---

#### `analyse_exam(exam, cls, source_row, datasets_dir, raw_nifti_dir) -> dict`

Loads every dataset variant for one exam and computes pad/crop metrics against the original FOV.

| Parameter | Type | Description |
|---|---|---|
| `exam` | `str` | Exam identifier (stem of the `.nii.gz` filename). |
| `cls` | `str` | Class subdirectory name (e.g. `"positive"`, `"negative"`). |
| `source_row` | `dict` | Row from `nifti_survey.csv` containing original dimensions, spacings, and FOV. |
| `datasets_dir` | `str` | Root directory for processed dataset folders. |
| `raw_nifti_dir` | `str` | Root directory for raw NIfTI files. |

**Returns:** `dict` — one wide row keyed by `exam`, original geometry fields (`orig_dim_*`, `orig_sp_*`, `orig_fov_*`), and per-dataset prefixed fields (`<ds_key>_shape`, `<ds_key>_sp_x/y/z`, `<ds_key>_pad_pct`, `<ds_key>_crop_pct`). Missing variant files produce empty strings for that dataset's metrics.

**Side effects:**
- Loads each NIfTI header (not the full voxel array) via `nibabel.load`; only `img.shape` and `img.header.get_zooms()` are accessed, so the image data array is not read into memory.
- Prints nothing; progress reporting is done by the caller.

---

#### `print_summary(rows) -> None`

Prints a formatted console summary of pad/crop percentages for every exam and per-dataset aggregates (median and max).

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Wide-format rows as returned by `analyse_exam`, one per exam. |

**Returns:** None.

**Side effects:**
- Writes a two-part table to stdout: per-exam `pad%/crop%` columns, then per-dataset median/max rows.
- Uses `numpy.median` and `numpy.max` for aggregate statistics.
- Skips empty-string metric values (missing variants) when computing aggregates.

---

#### `main(variants_csv, survey_csv, datasets_dir, raw_nifti_dir, output_csv) -> None`

Orchestrates the full pipeline: filters to available exams, analyses each one, writes `resize_quality.csv` in long format, and prints the summary.

| Parameter | Type | Description |
|---|---|---|
| `variants_csv` | `str` | Path to `nifti_survey_variants.csv`. |
| `survey_csv` | `str` | Path to `nifti_survey.csv`. |
| `datasets_dir` | `str` | Root directory for processed dataset folders. |
| `raw_nifti_dir` | `str` | Root directory for raw NIfTI files. |
| `output_csv` | `str` | Destination path for `resize_quality.csv`. |

**Returns:** None.

**Side effects:**
- Reads `variants_csv` and `survey_csv` into memory at startup (file handles opened without `with` blocks — rely on garbage collection for cleanup).
- Prints per-exam progress to stdout using `\r` overwrite.
- Converts the wide-format list of rows from `analyse_exam` into long format (one row per `(dataset, exam)`) in memory before writing.
- Writes (or overwrites) `resize_quality.csv` as a CSV with the columns defined in `FIELDNAMES`.
- Calls `sys.exit(1)` if no exam raw NIfTI files are found under `raw_nifti_dir`.
- Calls `print_summary` to emit the console summary after writing the file.

---

### Complexity flags

1. **Dual-metric symmetry not factored out.** `crop_pct` and `pad_pct` are near-mirrors of each other — both loop over three axes computing `min(covered, fov) / <denominator>` and then take the volumetric complement. A single helper `_axis_fracs(orig_fov, out_shape, out_spacing)` returning the three per-axis `min/covered` and `min/fov` ratios could serve both, eliminating the duplicated loop.

2. **Wide-to-long reshaping inside `main`.** The pivot from one wide dict per exam to one dict per `(dataset, exam)` is performed with nested loops and `dict.get` with empty-string defaults (lines 233–253). This is correct but verbose. A list comprehension or `itertools.product` over `(rows, DATASETS)` would be more concise and equally readable.

3. **`ds_key.split("_") if "_" in ds_key else (ds_key, "-")` inline in a loop.** The special-case for `"raw"` (which has no underscore) is handled with an inline conditional expression inside the long-format loop. Extracting this as a small helper or pre-building a `(variant, size)` lookup from `DATASETS` would separate the parsing concern from the pivot logic.

4. **Unclosed file handles in `main`.** `open(variants_csv)` and `open(survey_csv)` are passed directly to `csv.DictReader` without `with` blocks, identical to the pattern flagged in `check_fov_air.py`.

5. **`print_summary` determines the metric name from the stat label string.** The line `metric = "pad_pct" if "pad" in stat_label else "crop_pct"` (line 196) couples the display string to the data key via substring matching. Changing the label text could silently break metric selection. Using an explicit `(label, metric)` tuple in the `stat_fn` list would be safer.

## diagnostics/analyse_nifti_survey.py

### Purpose

Deep statistical analysis of `nifti_survey.csv` to inform the choice of target voxel shape, resampling strategy, and padding/cropping trade-offs before building any dataset. It is purely a decision-support tool — it writes no modified data, only console output and PNG plots.

---

### What it reads from `nifti_survey.csv`

Every row is consumed as a dict with the following fields:

| Field | Type after load | Meaning |
|---|---|---|
| `exam` | `str` | Exam identifier |
| `class` | `str` (`"0"` or `"1"`) | Label: 0 = healthy, 1 = aneurysm |
| `dim_x`, `dim_y`, `dim_z` | `int` | Voxel dimensions |
| `spacing_x_mm`, `spacing_y_mm`, `spacing_z_mm` | `float` | Voxel size in mm |
| `size_x_mm`, `size_y_mm`, `size_z_mm` | `float` | Physical FOV in mm (aliased internally as `fov_x/y/z`) |

The file is read once at startup via `load()` and the resulting list of dicts is passed to every subsequent function. No NIfTI files are opened.

---

### Decisions and insights the script is meant to support

- Whether to resample to a fixed isotropic resolution (e.g. 1 mm or 0.5 mm) or keep native spacing.
- Which target voxel shape (128³, 256³, 256×256×128, 256×256×192, etc.) minimises both padding waste and information-discarding crop, for every exam in the dataset.
- How severe the Z-axis anisotropy is across scanners, and whether it differs by class.
- What percentage of exams would fit inside each candidate shape without any cropping, at each candidate isotropic resolution.

---

### Plots produced

All plots are written to `<script_dir>/outputs/survey_plots/` (path built at module level from `__file__`; directory is created on import).

| Filename | Contents |
|---|---|
| `01_raw_properties.png` | 2×3 overlaid histogram grid: FOV X/Y/Z and spacing XY/Z distributions coloured by class, plus a Z/XY spacing-ratio histogram with an isotropic reference line. |
| `02_padding_vs_crop.png` | 2×3 scatter grid, one panel per key strategy: each point is one exam, x = padding %, y = cropped %, coloured by class, with 30 % padding and 10 % crop reference lines. |
| `03_fov_coverage.png` | Two line plots (0.5 mm iso and 1.0 mm iso): for each candidate target voxel count (64 … 384) the fraction of exams whose physical FOV fits without cropping, broken out per axis (X, Y, Z), with 90 % and 95 % reference lines. |

---

### Functions

#### `load(path) -> list[dict]`

Reads `nifti_survey.csv` and returns one dict per exam with integer voxel dimensions, float spacings, and float FOV values.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Absolute or relative path to `nifti_survey.csv`. |

**Returns:** `list[dict]` — one dict per row with keys `exam`, `class`, `dim_x/y/z`, `sx/sy/sz`, `fov_x/y/z`.

**Side effects:** Opens and reads one CSV file; no console output, no files written.

---

#### `col(rows, key) -> np.ndarray`

Extracts a single named field from a list of row dicts into a NumPy array.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Any subset of the loaded row list. |
| `key` | `str` | Dict key to extract (e.g. `"fov_x"`, `"sx"`). |

**Returns:** `np.ndarray` (1-D, dtype inferred from stored values).

**Side effects:** None.

---

#### `percentiles(arr, ps=(5, 25, 50, 75, 95)) -> dict`

Computes a set of percentile values for a 1-D array.

| Parameter | Type | Description |
|---|---|---|
| `arr` | `np.ndarray` | 1-D numeric array. |
| `ps` | `tuple[int, ...]` | Percentile levels to compute (default `(5, 25, 50, 75, 95)`). |

**Returns:** `dict` mapping each percentile integer to its `float` value.

**Side effects:** None.

---

#### `class_split(rows) -> tuple[list[dict], list[dict]]`

Partitions the full row list into the class-0 (healthy) and class-1 (aneurysm) subsets.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full or partial loaded row list. |

**Returns:** `(neg, pos)` — two `list[dict]` objects for `class == "0"` and `class == "1"` respectively.

**Side effects:** None.

---

#### `print_stat_block(label, arr) -> None`

Prints a single formatted statistics line (min, p5, p25, median, p75, p95, max) for one array to stdout.

| Parameter | Type | Description |
|---|---|---|
| `label` | `str` | Column/metric name used as a left-justified label. |
| `arr` | `np.ndarray` | 1-D numeric array to summarise. |

**Returns:** None.

**Side effects:** Writes one line to stdout.

---

#### `per_class_summary(rows) -> None`

Prints a full descriptive statistics block — total count, class counts, and per-field percentile tables — separately for class 0 and class 1.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full loaded row list. |

**Returns:** None.

**Side effects:** Writes approximately 20 lines to stdout covering voxel dimensions, spacings, and physical FOV statistics per class.

---

#### `spacing_anisotropy(rows) -> None`

Computes the per-exam Z/XY spacing ratio (`sz / sx`) and prints a binned ASCII bar chart plus median and p95 values.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full loaded row list. |

**Returns:** None.

**Side effects:** Writes approximately 10 lines to stdout. Fixed ratio bins are `[0, 1.1)`, `[1.1, 1.5)`, `[1.5, 2.0)`, `[2.0, 3.0)`, `[3.0, ∞)`.

---

#### `simulate_one(r, target_shape, resample_to_iso_mm, post_op) -> dict`

Simulates the full preprocessing path (optional resampling then crop or shrink-then-pad) for a single exam and returns waste metrics.

| Parameter | Type | Description |
|---|---|---|
| `r` | `dict` | One exam row with `fov_x/y/z`, `sx/sy/sz`, `dim_x/y/z`. |
| `target_shape` | `tuple[int, int, int]` | Target voxel shape `(tx, ty, tz)`. |
| `resample_to_iso_mm` | `float`, `"per"`, or `None` | `None` = native spacing; `float` = fixed isotropic resolution; `"per"` = resample each exam to its own minimum spacing. |
| `post_op` | `str` | `"crop"` = centre-crop to target (may discard data); `"shrink"` = scale so the largest dim fits, then zero-pad (never discards data). |

**Returns:** `dict` with keys `exam`, `class`, `fits` (`bool`), `cropped_frac` (`float` 0–1), `padding_frac` (`float` 0–1), `effective_spacing_mm` (`float`).

**Side effects:** None.

**Note on shrink logic:** the uniform scale factor is `min(tx/nx, ty/ny, tz/nz)`, capped at 1.0 to prevent upscaling. The effective spacing after shrink is `effective_spacing / scale`, reflecting the coarser resolution that results from fitting the volume into the target.

---

#### `simulate_strategy(rows, target_shape, resample_to_iso_mm, post_op) -> list[dict]`

Applies `simulate_one` to every exam in the dataset for a single strategy.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full loaded row list. |
| `target_shape` | `tuple[int, int, int]` | Forwarded to `simulate_one`. |
| `resample_to_iso_mm` | `float`, `"per"`, or `None` | Forwarded to `simulate_one`. |
| `post_op` | `str` | Forwarded to `simulate_one`. |

**Returns:** `list[dict]` — one result dict per exam (same order as `rows`).

**Side effects:** None.

---

#### `strategy_summary(name, sim_results) -> dict`

Aggregates per-exam simulation results into strategy-level scalar metrics.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable strategy label. |
| `sim_results` | `list[dict]` | Output of `simulate_strategy`. |

**Returns:** `dict` with keys `name`, `fits_n`, `needs_crop_n`, `fits_pct`, `median_pad_pct`, `p95_pad_pct`, `median_crop_pct`, `p95_crop_pct`, `heavy_pad_n` (exams with padding > 30 %), `heavy_crop_n` (exams with crop loss > 10 %).

**Side effects:** None.

---

#### `simulate_all(rows) -> dict`

Runs all 16 strategies defined in the module-level `STRATEGIES` list, prints a formatted comparison table, and returns all per-exam simulation results.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full loaded row list. |

**Returns:** `dict` mapping each strategy name (`str`) to its `list[dict]` of per-exam results.

**Side effects:** Writes an 18-row table (header + separator + one row per strategy) to stdout.

---

#### `plots(rows) -> None`

Generates and saves the three PNG files described in the "Plots produced" section above.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full loaded row list. |

**Returns:** None.

**Side effects:**
- Creates up to three matplotlib figures, saves them as PNGs to `PLOT_DIR`, and closes each figure.
- Prints the path of each saved file to stdout.
- Internally re-runs `simulate_strategy` for six key strategies to build the padding-vs-crop scatter data; results are not shared with `simulate_all`.

---

#### `coverage_table(rows) -> None`

Prints a tabular summary of what fraction of exams would fit without cropping on each axis, for target voxel counts of 128–320 at both 0.5 mm and 1.0 mm isotropic resampling.

| Parameter | Type | Description |
|---|---|---|
| `rows` | `list[dict]` | Full loaded row list. |

**Returns:** None.

**Side effects:** Writes approximately 20 lines to stdout (two blocks of six data rows each, one per isotropic resolution).

---

### Hardcoded paths and values

| Location | Value | Note |
|---|---|---|
| Module level | `CSV_PATH = <script_dir>/outputs/nifti_survey.csv` | Built from `__file__`; not configurable without editing the source. |
| Module level | `PLOT_DIR = <script_dir>/outputs/survey_plots` | Same; created automatically with `os.makedirs` at import time. |
| `plots()` figure title | `"NIfTI Survey — Raw Exam Properties (n=745)"` | The total exam count is hardcoded and will silently be wrong if the dataset grows or shrinks. |
| `STRATEGIES` list | 16 strategy tuples | Adding, removing, or renaming a strategy requires direct source edits. |
| `strategy_summary` | `> 0.30` for heavy padding, `> 0.10` for heavy crop | Alert thresholds are hardcoded. |
| `spacing_anisotropy` | Ratio bins `[0,1.1)`, `[1.1,1.5)`, `[1.5,2.0)`, `[2.0,3.0)`, `[3.0,99)` | Not configurable at runtime. |
| `coverage_table` | `isos = [0.5, 1.0]`, `targets = [128, 160, 192, 224, 256, 320]` | Both lists are hardcoded. |

---

### Complexity flags

1. **Silent missing scatter panels in `plots()`.** The `key_strategies` list inside `plots()` contains six `(internal_name, display_name)` pairs. Three of those internal names — `"256x256x192  iso-0.5mm"`, `"256x256x192  per-exam-iso  crop"`, and `"320x320x160  iso-0.5mm"` — do not appear in `STRATEGIES`. The inner lookup loop silently finds no match, so `all_sims_local` is never populated for those keys. The subsequent `sim = all_sims_local[internal_name]` access raises a `KeyError` at runtime, preventing `02_padding_vs_crop.png` from being written.

2. **`simulate_all` unpacks 4-element `STRATEGIES` tuples as 3 variables.** Each entry in `STRATEGIES` is `(name, shape, iso, post_op)`, but the loop in `simulate_all` unpacks only `name, shape, iso` and calls `simulate_strategy(rows, shape, iso)` — omitting the required `post_op` argument. This raises a `TypeError` at runtime. The `post_op` element is never consumed by `simulate_all`.

3. **`simulate_strategy` called a second time inside `plots()`.** Simulation is re-executed independently from `simulate_all`, doubling the computation for the six selected strategies without sharing results.

4. **Matplotlib backend forced at module import time.** `matplotlib.use("Agg")` is called unconditionally at the top level, which suppresses interactive display even when the script is imported into a notebook or interactive session.

5. **`os.makedirs(PLOT_DIR)` runs on import.** The output directory is created as a side effect of importing the module, not only when `plots()` is called.

## diagnostics/check_nifti.py and check_nifti_quality.py

### check_nifti.py

#### Overview

A minimal top-level script (no functions, no `main` guard) that spot-checks a single NIfTI file and prints basic statistics to stdout.

#### Execution model

There are no function definitions. All logic runs at module import time in the global scope.

| Step | What happens |
|---|---|
| Path selection | `sys.argv[1]` if provided; otherwise falls back to the hard-coded path `/mnt/data/cases-3/dataset_D_shrunk/1/BP001.nii.gz`. |
| Load | `nibabel.load(path)` followed by `img.get_fdata()` — loads the entire voxel array into memory as float64. |
| Print | Emits six lines to stdout: file path, shape, dtype, min, max, mean, and a non-zero voxel count with percentage. |

**Inputs:** One optional positional CLI argument — path to a `.nii.gz` file.

**Outputs:** Six formatted lines to stdout. No files written.

**Side effects:**
- Reads the full NIfTI volume into memory (no explicit release).
- Falls back silently to the hard-coded path if no argument is supplied; this path will fail on any other machine.

---

### check_nifti_quality.py

#### Overview

A more thorough diagnostic tool with a named function and `main` entry point. Inspects one file at a time and auto-discovers samples by scanning a hard-coded directory tree.

---

#### `check_nifti_file(nifti_path: str) -> None`

Performs a multi-section quality inspection of a single NIfTI file and optionally saves a slice plot.

| Parameter | Type | Description |
|---|---|---|
| `nifti_path` | `str` | Path to the `.nii.gz` file to inspect. |

**Returns:** None.

**Steps:** loads file, prints header metadata (shape, zooms, dtype, full affine), prints intensity stats (min, max, mean, NaN count, Inf count), saves a middle-axial-slice PNG to `/tmp/nifti_slice_<name>.png` if matplotlib is available.

**Side effects:** loads full float64 volume into memory (no release), writes PNG to `/tmp`, all output to stdout.

---

#### `main() -> None`

Auto-discovers one sample NIfTI from each of the two class subdirectories (`0/` and `1/`) under the hard-coded root `NIFTI_DIR = "/mnt/data/cases-3/nifti"`, then calls `check_nifti_file` on each.

**Side effects:** delegates all I/O to `check_nifti_file`. Prints directory-not-found warnings to stdout (not stderr).

---

### Comparison: overlap, differences, and merge potential

Both scripts serve the same purpose: load a NIfTI file and report basic properties to stdout. The fields in `check_nifti.py` are a strict subset of what `check_nifti_quality.py` reports.

| Dimension | check_nifti.py | check_nifti_quality.py |
|---|---|---|
| Target files | Single file via CLI arg | Auto-discovered first sample per class dir |
| Structure | Flat script, no functions | Module with `check_nifti_file` + `main` |
| Header fields | shape, dtype, min, max, mean, non-zero % | shape, zooms, dtype, affine, min, max, mean, NaN, Inf |
| Visualisation | None | Middle-slice PNG to `/tmp` |
| Hard-coded paths | Fallback path in argv branch | `NIFTI_DIR` constant |
| CLI | One positional arg | No CLI at all |
| `__main__` guard | Missing | Present |

**Merge recommendation:** Yes — a single script with `--file <path>` (single file) and `--dir <path>` (auto-discover per class) flags, plus a `--plot` option. The richer `check_nifti_quality.py` logic is the natural base.

---

### Complexity flags

1. **No `__main__` guard in `check_nifti.py`** — all logic runs at import time; importing in a test would immediately attempt to load a NIfTI file.
2. **Hard-coded paths in both files** — machine-specific, will fail anywhere else without editing source.
3. **`check_nifti_quality.py` has no CLI** — `main()` cannot be redirected without code changes.
4. **Warnings go to stdout, not stderr** in `check_nifti_quality.py::main`.
5. **Memory not released after `get_fdata()`** in both scripts — no `img.uncache()` or `gc.collect()`.

---

## diagnostics/compute_optimal_spacing.py

### Role in the pipeline

A standalone design-time decision tool. Reads no files at runtime — all dataset knowledge is hardcoded as module-level constants. Prints results to stdout. The output informs the choice of `spacing` passed to MONAI's `Spacingd` transform for fixed isotropic resampling experiments. Must be re-run whenever the dataset distribution or target grid shapes change.

---

### Module-level constants

| Constant | Type | Description |
|---|---|---|
| `GROUPS` | `list[tuple[int,int,int,int]]` | Four scanner/protocol archetypes encoding `(FOV_x_mm, FOV_y_mm, FOV_z_mm, exam_count)`. Covers 739 exams across four groups. BP082 is non-square in XY (163 mm x 180 mm). |
| `TOTAL` | `int` | Sum of all exam counts (739). Denominator for weighted averaging. |
| `TARGETS` | `dict[str, tuple[int,int,int]]` | Three candidate output grid shapes: `128x128x128`, `192x192x128`, `256x256x176`. |
| `CANDIDATES` | `list[tuple[str, Callable]]` | Six principled spacings reported alongside the grid-search optimum. |

Grid search runs over 0.40–1.795 mm in 0.005 mm steps (all hardcoded in `main`).

---

### Functions

#### `axis_score(fov, s, t) -> float`

Computes the efficiency score for a single axis given a candidate spacing.

| Parameter | Type | Description |
|---|---|---|
| `fov` | `float` | Field-of-view in mm for this axis. |
| `s` | `float` | Candidate isotropic voxel spacing in mm. |
| `t` | `int` | Target grid dimension in voxels. |

**Returns:** `float` in `(0, 1]` — computed as `min(r, t) / max(r, t)` where `r = fov / s`. A value of 1.0 means the resampled extent exactly fills the target; values below 1.0 indicate crop or pad. Score is symmetric: 10% crop and 10% pad yield the same value.

---

#### `weighted_efficiency(s, tx, ty, tz) -> float`

Exam-count-weighted mean volumetric efficiency across all scanner groups for a given spacing and target shape.

**Returns:** `float` in `(0, 1]`. Per-group score is the product of the three axis scores; the final value is the weighted average across `GROUPS` using exam counts as weights.

---

#### `per_group_scores(s, tx, ty, tz) -> list[float]`

Unweighted volumetric efficiency score per scanner group. Used only for display in `main`.

**Returns:** `list[float]` of length 4, one per group in `GROUPS`.

---

#### `main() -> None`

Grid search and reporting for every target shape, then prints a recommendation.

**Algorithm:**
1. For each shape in `TARGETS`, evaluates `weighted_efficiency` at every spacing in `np.arange(0.40, 1.80, 0.005)` (280 candidates). Selects the maximum as `best_s`.
2. Prints a neighbourhood table of rows within ±0.035 mm of `best_s`.
3. Prints the principled candidates table.
4. **Recommendation logic:** if the gap between grid-search optimum and the principled spacing `180/tx` is < 0.005, the principled spacing wins. Otherwise the grid-search optimum is used. Avoids choosing arbitrary-looking decimals when a physically motivated spacing is nearly as good.
5. For the recommended spacing, classifies each group-axis combination as crop/pad/exact (threshold: 0.5 voxels).

**Side effects:** stdout only. No files written.

---

### Complexity flags

1. **All dataset knowledge is hardcoded** — `GROUPS`, `TARGETS`, `CANDIDATES` are literals. Any dataset change requires editing source.
2. **Grid search step and bounds hardcoded** — `np.arange(0.40, 1.80, 0.005)` written directly in `main`.
3. **180 mm FOV constant appears in two places** — `CANDIDATES` lambda and recommendation logic; must be updated manually if dataset changes.
4. **Recommendation threshold 0.005 has no named constant or explanation**.
5. **`describe` closure redefined per loop iteration** — captures `rec_s` implicitly; harmless but inconsistent with the module's style.
6. **`axis_score` uses float approximation** — `r = fov / s` is a float; MONAI rounds to nearest int, so near half-integer boundaries the crop/pad classification may be off by one voxel.

---

## diagnostics/extract_gender.py and check_missing_gender.py

### extract_gender.py

**Purpose:** Extracts `PatientSex` from raw DICOM zip archives and writes the value back into `classes.csv`. This is a one-time enrichment script — it reads each zip, finds a DICOM header, and populates the `gender` column only where it is currently empty.

---

#### `normalise_exam_name(name) -> str | None`

Maps a raw folder name (e.g. `"bp001"`, `"BP_12"`) to canonical `"BP{NNN}"` format (zero-padded to 3 digits). Returns `None` if the name does not match `_BP_PATTERN`. No side effects.

---

#### `find_exam_folders(root) -> list[Path]`

Walks up to two directory levels under a temp extraction root and returns all directories matching the BP pattern. No side effects.

---

#### `find_first_dcm(folder) -> Path | None`

Walks `folder` recursively via `os.walk` and returns the path of the first `.dcm` file found, or `None`. Exits early on first hit. No side effects.

---

#### `read_patient_sex(dcm_path) -> str | None`

Opens a DICOM header only (`stop_before_pixels=True`), returns the `PatientSex` tag uppercased and stripped, or `None` on any failure. **Swallows all exceptions silently.**

---

#### `main() -> None`

Reads `classes.csv` in full, iterates over all `.zip` files in `/mnt/data/raw` (one at a time, extracting to a temp dir deleted in `finally`), reads one DICOM header per exam folder, accumulates results, then **overwrites `classes.csv`** if at least one new gender value was found.

**Columns touched in classes.csv:** reads all columns; conditionally appends the `gender` column header if absent; writes the `gender` cell only when the existing cell is empty and a fresh `PatientSex` in `{"M", "F"}` was read — **never overwrites a populated value.**

**Side effects:**
- Extracts each zip to a temp directory (cleaned up in `finally`).
- Overwrites `classes.csv` when new gender data is found.
- Hardcoded zip root: `/mnt/data/raw` (machine-specific).

---

### check_missing_gender.py

**Purpose:** A 12-line diagnostic that reads `classes.csv` and `nifti_survey.csv` via pandas, joins on `exam`, and prints counts and a table of rows where `gender` is not in `{0, 1}`. Writes nothing.

No named functions — all logic runs at module level.

**Inputs:** `dataset/classes.csv`, `diagnostics/outputs/nifti_survey.csv` (paths relative to `BASE = Path(__file__).resolve().parents[2]`).

**Outputs:** Three printed counts and a table of offending rows to stdout.

**Flag:** Uses `pandas` for a join + filter that stdlib `csv` could handle with equal clarity and no extra dependency.

---

### Overlap and merge assessment

| Dimension | extract_gender.py | check_missing_gender.py |
|---|---|---|
| Reads classes.csv | Yes | Yes |
| Reads DICOM zips | Yes | No |
| Reads nifti_survey.csv | No | Yes |
| Writes anything | Yes (classes.csv) | No |
| Purpose | Populate missing gender values | Report which exams still lack gender |

The two scripts are complementary steps of the same workflow (first populate, then verify) but have no overlapping logic. **They should not be merged** — they have different heavy dependencies and run at different points. However, `check_missing_gender.py` is a candidate for deletion: its output (`inventory_scan.py`'s `_log_status_breakdown` already reports unlabelled gender counts) is largely superseded.

---

### Flags

1. **`read_patient_sex` swallows all exceptions silently** — any DICOM read failure (corrupt file, missing tag, encoding error) returns `None` with no log entry, making it impossible to distinguish "no sex tag" from "file unreadable".
2. **Hardcoded zip root `/mnt/data/raw`** in `main` — machine-specific; fails anywhere else.
3. **`check_missing_gender.py` uses pandas for a 3-line join** — replaceable with stdlib `csv` + a set lookup.
4. **`check_missing_gender.py` has no `__main__` guard** — all logic runs at import.
5. **`extract_gender.py` functionality is superseded** by `inventory_scan.write_gender_to_classes_csv`, which performs the same DICOM sex extraction and write-back using the already-validated case list rather than raw zips.
