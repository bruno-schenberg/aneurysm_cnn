# Data Engine Requirements

## 1. Data Quality

### 1.1 DICOM Ingestion

- Accept a raw folder of per-case DICOM subdirectories; traverse and rename folders to a canonical `BP{NNN}` scheme.
- Validate each series: check geometric consistency (uniform slice spacing, no gaps), orientation (`get_orientation` → axial/coronal/sagittal), modality presence, and slice-thickness uniformity.
- Detect and flag mixed-series folders (multiple UIDs in one folder); report per-series breakdown in `mixed_series_analysis.csv` (this file stays separate — it is per-series, not per-case).
- Filter scout slices (`ImageType` must contain `'ORIGINAL'`); record `scout_slice_count` per case.
- Flag geometric outliers by exam size (`_flag_exam_size_outliers`); include `exam_size` in the per-case report.
- Attach ground-truth label (`class`, `location`) from `classes.csv` via suffix-stripped exam-name lookup; mark unlabelled valid cases as `MISSING_CLASS`.
- Produce a single **`pipeline_report.csv`** (one row per input case, 19 columns) that merges all per-case fields currently split across `folder_rename_map.csv` and `ingestion_summary.csv`:
  - `original_name`, `fixed_name`, `data_path`, `total_dcms`
  - `issues` (pipe-separated list of all flags, e.g. `VARIABLE_SPACING|MISSING_CLASS`), `duplicate_slice_count`, `scout_slice_count`
  - `orientation`, `modality`, `slice_thickness`, `patient_age`, `patient_sex`
  - `image_dimensions`, `class`, `location`, `exam_size`
  - `conversion_status`, `conversion_reason`, `output_path`
- All issues for a case must be accumulated, not overwritten. If a case has multiple problems (e.g. `VARIABLE_SPACING` and `MISSING_CLASS`), all must appear in the `issues` field; no issue may silently replace another.
- Cases that never reached conversion must appear in `pipeline_report.csv` with `conversion_status = "not_attempted"` and `conversion_reason` derived from the `issues` field.
- `folder_rename_map.csv` and `ingestion_summary.csv` are retired once `pipeline_report.csv` is in place.

### 1.2 Diagnostics Consolidation

Current diagnostics scripts (9 total) and their disposition:

| Script | Status | Merged subcommand |
|---|---|---|
| `survey_nifti.py` | Merge | `diagnose.py survey` |
| `analyse_nifti_survey.py` | Merge | `diagnose.py analyse` |
| `check_nifti.py` | Merge | `diagnose.py check FILE` |
| `check_nifti_quality.py` | Merge | `diagnose.py quality FILE` |
| `check_resize_quality.py` | Merge | `diagnose.py resize` |
| `check_fov_air.py` | Merge | `diagnose.py fov-air` |
| `compute_optimal_spacing.py` | Merge | `diagnose.py spacing` |
| `extract_gender.py` | **Delete** — superseded by `inventory_scan.write_gender_to_classes_csv` |
| `check_missing_gender.py` | **Delete** — superseded by `inventory_scan._log_status_breakdown` |

- The 7 surviving scripts merge into `data_engine/diagnostics/diagnose.py` under an `argparse` subcommand interface; all share `nibabel`/`numpy` and operate on the same `nifti/` output directory.
- `diagnose.py survey` runs first; `analyse`, `resize`, `fov-air`, and `spacing` depend on its `nifti_survey.csv` output.
- All diagnostic output goes to `data_engine/diagnostics/outputs/`; no diagnostic script writes to the main pipeline CSVs.
- Deleting `check_missing_gender.py` removes the only `pandas` dependency in the diagnostics directory.
- **Artifact budget: the entire data quality step must produce at most 3 output files:** `pipeline_report.csv` (per-case ingestion + conversion), `diagnostics_report.csv` (all post-conversion checks merged into one row per case), and `mixed_series_analysis.csv` (stays separate — it is per-series, not per-case). No other CSVs, no JSONL logs.
- See `docs/de_diagnostics.md` for per-script detail.

### 1.3 NIfTI Conversion

- Convert validated DICOM series to NIfTI (`.nii.gz`) using ITKReader; preserve the affine matrix; cast to float32.
- Write NIfTI files to `--nifti-dir`; guard against writes to local disk when the target filesystem is unmounted.
- After successful NIfTI conversion, discard raw DICOM source data to recover disk space (do not store both).
- Record `conversion_status` (`success` / `skipped` / `failed`) and `output_path` per case in `pipeline_report.csv`.
- Only cases with `validation_status == 'OK'` and a valid class label proceed to conversion (`filter_for_conversion`).

### 1.4 Incremental Ingestion

- Support a `--incremental` (or equivalent) mode that:
  - Reads the existing `nifti/` directory to build a set of already-converted case names.
  - Scans a secondary raw input folder (e.g. `raw2/`) for new cases not present in that set.
  - Runs the full ingestion pipeline only on the new cases; skips all existing ones.
  - Appends new rows to `pipeline_report.csv` rather than rewriting it.
- No re-processing of existing NIfTI files; no modification of already-converted data.

### 1.5 Cases Grouping

- `diagnose.py survey` must identify all unique acquisition configurations (distinct `(dim_x, dim_y, dim_z, spacing_x, spacing_y, spacing_z)` tuples) present in the NIfTI dataset and cluster them into scanner-protocol groups using a greedy tolerance-based algorithm (XY matrix match + 15% spacing_x / 20% spacing_z thresholds).
- Groups are the primary unit for dataset-level analysis: FOV coverage, air depth, and resize quality checks all operate per-group.
- The survey output must record `group_id` and `group_count` per case so any downstream script can filter or stratify by scanner group without re-running the survey.
- See `docs/data_engine/de_diagnostics.md` → `survey_nifti.py` for the current grouping algorithm.

## 2. Dataset Generation

### 2.1 Dataset Approaches

- Six preprocessing variants (A–F) must be supported, forming a 3×2 factorial design:
  - **Crop axis:** 80% centre-crop (A, C, E) vs. full volume (B, D, F).
  - **Resampling strategy:** no resampling / trilinear resize (C, D); per-scan dynamic isotropic spacing (A, B); fixed dataset-wide isotropic spacing (E, F).
- Three output resolutions must be supported: `128×128×128`, `192×192×128`, `256×256×176`; each variant can be generated at any resolution independently.
- Input files must be sourced from the NIfTI output of `data_cleaner.py`, organised into `0/` and `1/` class subdirectories.
- Output directory layout mirrors input class structure; filenames are preserved; one output directory per (variant, resolution) combination.
- Fixed-spacing variants (E, F) derive their spacing from dataset-wide FOV maxima (192 mm for cropped, 240 mm for full); these constants must be updated if the dataset grows to include larger FOVs.

### 2.2 Local vs. Cluster

- Full datasets (all variants, all cases) are generated exclusively on the cluster via SLURM (`dataset_gen.slurm` / `dataset_gen_128.slurm`).
- Local disk holds only small sample datasets (a subset of cases) for development and smoke-testing; sample datasets use the same variant pipeline with the same parameters.
- Sample datasets must include at least one case from each scanner-protocol group (see §1.5) so all acquisition types are exercised during local testing.
- Sample datasets must be reproducible from the same NIfTI input files and the same CLI arguments.

### 2.3 Dataset Versioning and Traceability

- Each generated dataset directory must be traceable to: the set of input NIfTI files that produced it, the variant key, the target resolution, and the `--seed` used for file-list shuffling.
- If a dataset directory is regenerated with different parameters, the old one must not be silently overwritten; versioning or namespacing by parameters is required.
- Processing failures (per-file errors) must be logged and surfaced as a non-zero exit code; no silent partial datasets.

### 2.4 Reproducibility

- The file-list shuffle seed (`--seed`, default `42`) must be recorded alongside the dataset so any run can be reproduced exactly.
- Worker-level processing is deterministic given the same input file and variant; no random transforms are applied during dataset generation.
- Idempotency: re-running `dataset_gen.py` on a completed dataset skips already-written files without reprocessing them.
