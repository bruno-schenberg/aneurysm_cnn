# Data Engine

Design document for the data pipeline: raw DICOM archives to validated NIfTI to resampled training datasets. This covers design decisions, QC internals, and variant rationale. For the step-by-step pipeline walkthrough, see [`PIPELINE.md`](../PIPELINE.md).

## Guiding constraints

- One orchestrator script (`run_pipeline.py`) drives all steps.
- Steps are independent packages under `steps/`; zero cross-step dependencies.
- Each step reads the previous step's CSV --- observable inputs and outputs.
- Raw data is never deleted or modified.
- At most 3 pipeline output CSVs; two in the common case.
- Dataset generation (resampling) is a separate step, not part of conversion.
- `config.py` is the single source for paths, thresholds, and variant definitions.

---

## Module layout

`data_engine` is a thin, project-specific package: archive extraction, exam
discovery, DICOM->NIfTI conversion, resampling, and MIP generation are
implemented in the [`medprep`](https://github.com/bruno-schenberg/medprep)
library and imported here, not reimplemented. What lives in this repo is the
orchestration, the project-specific naming/QC-tier rules, and config.

```
data_engine/
├── pyproject.toml         # package def; depends on medprep
├── config.py              # paths, thresholds, VARIANTS, enable_test_mode()
├── scripts/
│   ├── run_pipeline.py    # orchestrator (--step, --test, --dry-run, --variants)
│   ├── make_sample.py     # build sample_dataset/ for local testing
│   └── justification/     # research scripts (centroid crop, cube shape, isotropy)
├── steps/
│   ├── step0/              # discover + name (project wrapper; extraction/discovery
│   │   │                   # themselves are medprep.discovery calls, see PIPELINE.md)
│   │   ├── naming.py        # wraps medprep.ids.assign_sequential_ids (BP#### regex)
│   │   └── step0_discover.py
│   ├── step1/              # metadata join + QC + conversion + grouping
│   │   ├── metadata.py      # wraps medprep.metadata.join_metadata
│   │   ├── qc.py             # GE MRA series-selection tiers on top of medprep.qc_checks
│   │   ├── converter.py     # calls medprep.nifti_convert
│   │   ├── grouping.py      # project-specific resolution grouping, no medprep call
│   │   └── step1_convert.py
│   ├── step2/               # dataset variant generation
│   │   └── step2_resample.py  # calls medprep.resample.run_resampling
│   ├── step3/                # visual QC
│   │   └── step3_visualize.py # uses medprep.visualize
│   └── step_mip/              # 2D representation generation
│       └── generate_*.py      # mostly thin wrappers over medprep.mip;
│                               # generate_frangi_mips.py is project-specific (kept out of medprep)
└── reports/                # pipeline CSV outputs (gitignored)
```

Detailed pipeline walkthrough (what's a medprep call vs. project code, per
step): [`PIPELINE.md`](../PIPELINE.md).

---

## Output files

| File | Grain | When written |
|------|-------|--------------|
| `step0_discover.csv` | one row per exam (incl. gap rows) | step 0 |
| `step1_convert.csv` | one row per exam (full pipeline report) | step 1 |
| `mixed_series_report.csv` | one row per series in mixed-series folders | step 1, if mixed-series exams exist |
| `step2_resample.csv` | one row per (exam, variant) | step 2 |
| `step3_visualize.png` | one figure | step 3 |

Column accumulation is append-only: each stage adds columns, never mutates a previous stage's. The single derived `qc_block_reason` column (step 1) is the human- and machine-readable summary of why an exam is or isn't eligible for conversion.

---

## QC design

### Pre-QC gate

QC runs on exams with `folder_status == ready`, `name_parse_status` not in {unparseable, gap}, and `class` not in `config.EXCLUDE_CLASSES`. Missing gender/age/class exams are still QC-checked --- their DICOM quality is worth recording; metadata gaps are a conversion-gate concern, not a QC one.

### QC columns

QC was deliberately collapsed from a per-symptom flag set into a small number of grouped columns. The underlying truth for most blocking cases is the same --- the exam is unusable and needs a fresh PACS export --- so granular flags added no value downstream. Every consumer checks one thing: `qc_block_reason is None`.

Blocking columns (`_QC_BLOCK_COLUMNS`):

| Column | Triggers when |
|--------|---------------|
| `corrupt_headers` | every DICOM header fails to parse |
| `missing_spatial_metadata` | IOP, IPP, or PixelSpacing absent from a slice |
| `slice_issue` | no ORIGINAL slices, single slice, duplicate slices (gap < 0.001 mm), or missing slice (gap > 1.5x median) |
| `inconsistent_spacing` | std of inter-slice gaps > `SPACING_STD_TOLERANCE_MM` (0.1 mm) |
| `extent_issue` | z-extent < 90 mm or > 300 mm |
| `no_valid_series` | mixed-series exam with no axial-3D candidate |

Non-blocking info columns: `scout_slices` (scouts were present and filtered), `mixed_series` (folder held multiple SeriesInstanceUIDs), `z_order_differs` (IPP sort order differs from InstanceNumber order).

### qc_block_reason

A single derived column consolidating **all** conversion blockers --- structural, metadata, and DICOM QC --- as a comma-separated string, or `None` if the exam is fully eligible. Values: `not_ready`, `unparseable`, `no_metadata_match`, `no_class`, `excluded_class`, `missing_patient_data`, plus the six QC columns above. `missing_patient_data` fires when gender **or** age is null: both are required so that "with metadata" and "without metadata" training runs use the same cohort.

### Metrics recorded

`slice_count`, `scout_slice_count`, `modality`, `orientation`, `rows`, `columns`, `voxel_spacing_row`, `voxel_spacing_col`, `voxel_spacing_z`, `extent_z_mm`, `series_number`, `series_description`, `protocol_name`, `mra_acquisition_type`, `scanning_sequence`, `scan_options`, `selected_series_uid`.

`extent_z_mm` is `positions[-1] - positions[0]` --- the physical span between the first and last slice along the acquisition direction, not `slice_count x z_spacing` (which would include one extra interval). Positions come from projecting `ImagePositionPatient` onto the slice normal (IOP cross product), so the value is correct for oblique acquisitions.

---

## Mixed-series selection

When a folder holds multiple SeriesInstanceUIDs, every series is characterised with the same QC pipeline, then one is selected:

1. **Candidate filter** (all must hold): `orientation == axial`, `mra_acquisition_type == 3D`, `slice_count > 1`. This drops reformats, localizers, 2D structural MRI, and single-image MIPs.
2. **Tier ranking** among candidates:
   - Tier 1: `VASCTOF_GEMS` present, `FILTERED_GEMS` absent --- primary TOF MRA off the scanner.
   - Tier 2: both present --- filtered reconstruction; used only when no unfiltered version exists.
   - Tier 3: neither tag (e.g. SAT1 scanner generation) --- primary TOF MRA on older GE hardware.

   Within a tier, most slices wins. If no candidate survives the filter, the exam is blocked with `no_valid_series` --- there is no silent fallback to an unusable series.

Every series (selected or not) is written to `mixed_series_report.csv` with a `selected` boolean, so diagnostics are fully covered without picking a fake primary.

Tag definitions: `VASCTOF_GEMS` --- GE ScanOptions marker for Vascular TOF MRA; `FILTERED_GEMS` --- post-processed filtered reconstruction (always derivative); `SAT1` --- venous saturation pulse used on the 768-matrix generation (mostly single-series folders).

---

## Centroid crop (180 mm)

Immediately after ITKReader produces the numpy volume and before nibabel saves it, each XY axis is cropped or padded to exactly `TARGET_FOV_MM = 180 mm`. The raw DICOMs are never touched.

**Why.** Native FOV varies by acquisition group (SAT1 180 mm, VASCTOF up to 240 mm). Saving at native FOV forced the resampler to use a 240 mm ceiling so the widest exams weren't clipped, which penalised the 65% of exams already at 180 mm with a coarser isotropic spacing. Cropping every exam to a uniform 180 mm XY FOV at conversion time removes that: the whole dataset is homogeneous and the resampler only handles Z.

**Why not a geometric centre crop.** Patient heads sit systematically below the image centre in VASCTOF acquisitions, so a geometric-centre crop wastes voxels asymmetrically. Instead the crop window is placed on the **intensity-weighted XY centroid** of the bright tissue, computed from a mean projection of the middle 50% of slices (thresholded at the p20 of non-zero voxels to exclude air). The same XY offset is applied to every slice.

**Why 180 mm.** It is the minimum FOV in the dataset --- SAT1 (43%) and the 163x180 mm group (22%) are already at or below it, so they pass through untouched; VASCTOF exams have ~30 mm of air margin per side that the crop removes.

Per-axis behaviour: oversized -> crop centred on centroid (clamped to the image boundary, never padded outward); undersized -> symmetric zero-pad; exact -> unchanged. The affine origin is shifted to keep the NIfTI physically correct, and `crop_x0` / `crop_y0` are recorded so the crop is auditable.

---

## Dataset variant rationale

### Two isotropic cubes

Because every NIfTI is a uniform 180 mm XY FOV after the centroid crop, spacing is simply `180 / dim`:

| Variant | Shape | Spacing (mm/voxel) |
|---------|-------|--------------------|
| `iso_128` | 128 x 128 x 128 | 1.40625 |
| `iso_256` | 256 x 256 x 256 | 0.703125 |

A non-isotropic 128 control variant (finer Z than XY) is planned; it is stubbed out in `config.VARIANTS` pending a chosen Z spacing.

### Design decisions

- **No crop variants.** Earlier designs had a 3x2x3 factorial (18 datasets). Centre cropping is now handled by random-zoom augmentation at training time (90--110% zoom in the MONAI transform stack), which gives the same regularisation without materialising extra datasets on disk. Per-exam dynamic-spacing strategies were also dropped: they add complexity with no benefit once every exam shares a uniform 180 mm XY FOV.
- **192 dropped.** Only 128 and 256 are generated.
- **Cubic Z, always padded.** The dataset's physical Z extent (median ~135 mm, max 167 mm) is always shorter than the 180 mm XY FOV, so Z fill is 65--93% at any cubic size and the remainder is honest air/skull-boundary padding --- Z is never upsampled. The nearest divisible-by-32 shape that covers the max Z extent happens to equal the XY voxel count, so the volumes are numerically 128 cubed and 256 cubed. A tighter non-cube (e.g. x96) isn't divisible by 32 and would break UNet/UNETR; 128 is the next safe size, giving back the cube.
- All standard model sizes (128, 160, 192, 224, 256) are multiples of 32, which is safe for every architecture in `training_engine`. ResNets accept any size (adaptive pooling); UNet3D and UNETR require divisible-by-16.

### Resampling strategy

Strategy `iso_fixed`: resample to a fixed isotropic spacing (trilinear / `scipy.ndimage.zoom` order 1), then symmetric pad/crop to the target shape. One process per file; idempotent (existing outputs skipped).

---

## Test mode

`config.enable_test_mode()` redirects all data paths to `sample_dataset/`, a small self-contained copy of the pipeline's inputs. Built by `scripts/make_sample.py`. Activated via `run_pipeline.py --test`.

---

## Key design decisions

- **Single orchestrator, `--dry-run` for audit.** No separate inventory pass.
- **Raw data is read-only.** The pipeline extracts and reads; cleanup is manual.
- **One `qc_block_reason` column** drives conversion eligibility --- consumers never test individual QC columns.
- **Mixed-series gets its own CSV** because its grain (one row per series) differs from the per-exam report.
- **Centroid crop at conversion** makes the on-disk NIfTI dataset homogeneous (uniform 180 mm XY FOV), which is what lets the resampler stay simple.
- **Step scripts are plain functions**, not standalone CLIs --- called by the orchestrator.
- **Script name = CSV name** consistency (step0_discover.py produces step0_discover.csv).
