"""Stage 5 — DICOM quality checks and mixed-series selection.

Mixed-series selection overview
--------------------------------
Some exam folders on disk contain DICOM files from multiple distinct
acquisitions mixed together (identified by SeriesInstanceUID). Before QC
checks can run we must select one series to represent the exam.

The selection is automatic and documented in full in _select_primary_series.
All series in a mixed-series folder are characterised and written to
mixed_series_report.csv so every decision is auditable.

Flow (per-exam)
---------------
1. _check_exam loads all DICOM headers from the exam folder.
2. Groups files by SeriesInstanceUID.
3. Single series  → _check_series runs directly.
4. Multiple series → every series is characterised by _check_series,
   the primary is selected via _select_primary_series, the selected
   series' QC becomes the exam-level result, and all series are
   recorded in mixed_rows for the mixed_series_report.

The per-check primitives (spatial metadata, spacing, extent, scout
filtering, series grouping) are scanner-agnostic and live in
medprep.qc_checks. Everything below is specific to this project: the
GE MRA series-selection tiers (VASCTOF_GEMS/FILTERED_GEMS/SAT1) and the
block-reason policy.
"""
from pathlib import Path

import pandas as pd

import config
from medprep.dicom_io import load_headers
from medprep.qc_checks import (
    group_by_series_uid,
    read_series_tags,
    filter_scouts,
    check_spatial_metadata,
    sorted_projected_positions,
    project_position,
    check_spacing,
    check_extent,
)

# Blocking QC columns — any truthy value means the exam cannot be converted.
#
# corrupt_headers: DICOM headers could not be read at all (pydicom parse error).
#   Note: reads headers only (stop_before_pixels=True). Files with valid headers
#   but corrupted pixel data are not caught here — those fail at conversion time.
#
# missing_spatial_metadata: IOP/IPP/PixelSpacing absent or malformed. Irrecoverable
#   — this is a scanner metadata issue, not a PACS export problem. Re-upload won't fix it.
#
# slice_issue: problem with the slice set itself. Values:
#   "no_originals"   — all slices are DERIVED (no ORIGINAL ImageType)
#   "single_slice"   — only one original slice
#   "duplicates"     — two or more slices at the same z-position
#   "missing_slices" — gap > 1.5× the median inter-slice spacing
#   (multiple issues can be comma-separated if they co-occur)
#
# inconsistent_spacing: spacing standard deviation exceeds threshold. Irrecoverable
#   — not a PACS/re-upload issue. Worth investigating if it appears in the dataset.
#
# extent_issue: z-extent outside acceptable range.
#   "too_short" — extent < EXTENT_MIN_MM
#   "too_long"  — extent > EXTENT_MAX_MM
_QC_BLOCK_COLUMNS = [
    "corrupt_headers",
    "missing_spatial_metadata",
    "slice_issue",
    "inconsistent_spacing",
    "extent_issue",
    "no_valid_series",  # mixed-series exam with no axial 3D candidate
]

# Non-blocking informational columns.
_QC_INFO_COLUMNS = [
    "scout_slices",  # derived/scout files were present and filtered out
    "mixed_series",  # folder contained multiple SeriesInstanceUIDs
    "z_order_differs",  # projection-based sort differs from raw Z sort
]

_QC_METRICS = [
    "slice_count",
    "scout_slice_count",
    "modality",
    "orientation",
    "rows",
    "columns",
    "voxel_spacing_row",    # PixelSpacing[0]: mm between rows  (y-axis)
    "voxel_spacing_col",    # PixelSpacing[1]: mm between columns (x-axis)
    "voxel_spacing_z",
    "extent_z_mm",          # physical z-span: positions[-1] - positions[0]
    "series_number",
    "series_description",
    "protocol_name",
    "mra_acquisition_type",
    "scanning_sequence",
    "scan_options",
    "selected_series_uid",  # UID of the series used; None for single-series exams
]

_ALL_QC_COLUMNS = _QC_BLOCK_COLUMNS + _QC_INFO_COLUMNS + _QC_METRICS

# Columns written per-series to mixed_series_report.csv.
# Includes a `selected` boolean so every audit can trace which series was used.
_MIXED_SERIES_COLUMNS = (
    ["exam_id", "series_uid", "selected"]
    + [c for c in _QC_BLOCK_COLUMNS if c != "corrupt_headers"]
    + ["scout_slices"]
    + [m for m in _QC_METRICS if m != "selected_series_uid"]
)


# ---------------------------------------------------------------------------
# GE MRA series selection (project-specific)
# ---------------------------------------------------------------------------

def _has_scan_option(series_summary: dict, tag: str) -> bool:
    """True if tag appears in the series' scan_options string."""
    opts = series_summary.get("scan_options")
    return tag in str(opts) if opts else False


def _select_primary_series(series_summaries: list[dict]) -> str:
    """
    Select the SeriesInstanceUID of the primary MRA acquisition from a
    mixed-series folder.

    Background
    ----------
    Some exam folders contain DICOM files from multiple acquisitions.
    This happens because the scanner or PACS exported the full study
    rather than a single series, or because different reconstructions
    of the same acquisition were stored in the same directory.

    All series are characterised by _check_series before this function
    is called, so the full QC result dict is available for each series.

    Filter step — candidate series
    --------------------------------
    A series is a candidate only if ALL three conditions hold:

      orientation == "axial"
          Removes sagittal and coronal reformats, multi-planar
          reconstructions, and planning localizer series.

      mra_acquisition_type == "3D"
          Removes 2D structural MRI sequences (Spin Echo, Inversion
          Recovery, etc.) that appear in the same study.  The target
          acquisition for this project is always a 3D Time-of-Flight
          MRA volume.

      slice_count > 1
          Removes single-image MIP (Maximum Intensity Projection)
          projections, which are derived summary images saved alongside
          the 3D volume.  These have ImageType = DERIVED but can
          occasionally slip through if ImageType is absent.

    After this filter, the current dataset yields exactly one candidate
    per mixed-series exam in all 17 cases (verified manually — see Tier
    system below).

    Tier system — ranking among candidates
    ----------------------------------------
    When multiple candidates survive the filter, the following priority
    applies.  Only the highest applicable tier is used; within a tier
    the series with the most slices wins.

    Tier 1 — VASCTOF_GEMS present, FILTERED_GEMS absent
        VASCTOF_GEMS is a GE ScanOptions tag that marks the primary
        Vascular Time-of-Flight MRA acquisition.  When it appears
        without FILTERED_GEMS the series is the raw, unprocessed
        acquisition off the scanner.  This is the preferred source
        for conversion.

    Tier 2 — VASCTOF_GEMS present, FILTERED_GEMS also present
        FILTERED_GEMS is a GE ScanOptions tag meaning the scanner
        applied a spatial smoothing filter and saved the result as a
        separate series.  It is always derivative of another series.
        When the only VASCTOF series in the folder also has FILTERED_GEMS
        (no unfiltered alternative exists), this series is still the
        correct MRA acquisition — it is equivalent to the 128 single-
        series exams in the dataset that also carry VASCTOF+FILTERED.

        Dataset edge cases: BP0178 and BP0319 fall in this tier.
        Their folders contain no unfiltered VASCTOF series.
        SeriesDescription and ProtocolName were stripped during
        anonymisation, so these two were verified by confirming that:
          (a) no axial 3D VASCTOF-without-FILTERED series exists,
          (b) the selected series has the same spacing and matrix as the
              other 128 VASCTOF+FILTERED single-series exams, and
          (c) the other series in these folders are either filtered
              reconstructions of non-TOF sequences or localizers.

    Tier 3 — VASCTOF_GEMS absent (e.g. SAT1 scanner generation)
        539 exams in the dataset come from a different GE hardware/
        software generation that does not emit the VASCTOF_GEMS tag.
        These exams use a venous saturation pulse (ScanOptions = SAT1)
        instead.  For mixed-series folders from this scanner generation
        the candidate filter leaves only the primary acquisition, so
        Tier 3 is the "no VASCTOF tag" fallback — pick the axial 3D
        candidate with the most slices.

    Fallback — no candidate passed the filter
        If no axial 3D series with slice_count > 1 exists (should not
        happen with the current dataset), fall back to the series with
        the most slices, preferring axial orientation.

    Verification summary
    ---------------------
    The algorithm was validated against all 17 mixed-series exams
    (252 total series, 32 axial-3D candidates after filtering).
    Every exam resolves to exactly one series.  All 17 selected series
    map to existing acquisition groups in the dataset — none are outliers.

    Parameters
    ----------
    series_summaries : list[dict]
        One dict per series, keyed by _ALL_QC_COLUMNS + "series_uid".
        Produced by _check_series for every series in the folder.

    Returns
    -------
    str — the SeriesInstanceUID of the selected series.
    """
    def is_candidate(s: dict) -> bool:
        return (
            s.get("orientation") == "axial"
            and str(s.get("mra_acquisition_type") or "").upper().strip() == "3D"
            and (s.get("slice_count") or 0) > 1
        )

    candidates = [s for s in series_summaries if is_candidate(s)]

    if not candidates:
        return None  # no axial 3D series — exam blocked as no_valid_series

    # Tier 1: primary TOF acquisition (VASCTOF, unfiltered)
    tier1 = [s for s in candidates
             if _has_scan_option(s, "VASCTOF_GEMS")
             and not _has_scan_option(s, "FILTERED_GEMS")]
    if tier1:
        return max(tier1, key=lambda s: s.get("slice_count") or 0)["series_uid"]

    # Tier 2: filtered TOF acquisition (VASCTOF + FILTERED — no unfiltered available)
    tier2 = [s for s in candidates if _has_scan_option(s, "VASCTOF_GEMS")]
    if tier2:
        return max(tier2, key=lambda s: s.get("slice_count") or 0)["series_uid"]

    # Tier 3: no VASCTOF tag (SAT1 scanner generation or other)
    return max(candidates, key=lambda s: s.get("slice_count") or 0)["series_uid"]


# ---------------------------------------------------------------------------
# Series-level orchestrator
# ---------------------------------------------------------------------------

def _check_series(datasets: list) -> dict:
    """
    Run all QC checks on one series (a pre-loaded list of DICOM datasets).
    Returns a result dict for all QC columns except corrupt_headers and
    mixed_series (those are set at the exam level by _check_exam).

    Checks run in order; blocking failures return early so downstream checks
    that depend on valid data are never executed on bad input.
    """
    result = {col: None for col in _ALL_QC_COLUMNS}

    original, scout_fields = filter_scouts(datasets)
    result.update(scout_fields)
    if not original:
        return result  # slice_issue == "no_originals"

    result.update(read_series_tags(original))

    if len(original) == 1:
        result["slice_issue"] = "single_slice"
        return result

    spatial_fields, normal = check_spatial_metadata(original)
    result.update(spatial_fields)
    if normal is None:
        return result  # missing_spatial_metadata == True

    positions = sorted_projected_positions(original, normal)
    if positions is None:
        result["missing_spatial_metadata"] = True
        return result

    # Check whether raw Z ordering differs from projection-based ordering.
    try:
        raw_z = [float(ds.ImagePositionPatient[2]) for ds in original]
        rank_z    = sorted(range(len(raw_z)), key=lambda k: raw_z[k])
        rank_proj = sorted(range(len(original)),
                           key=lambda k: project_position(original[k], normal))
        result["z_order_differs"] = rank_z != rank_proj
    except (AttributeError, IndexError, ValueError):
        pass

    result.update(check_spacing(positions, config.SPACING_STD_TOLERANCE_MM))
    result.update(check_extent(positions, config.EXTENT_MIN_MM, config.EXTENT_MAX_MM))

    return result


def _check_exam(dcm_path: str) -> tuple[dict, list[dict]]:
    """
    Load headers, detect mixed series, select primary series, run QC.

    For single-series exams: runs QC directly.
    For mixed-series exams:  characterises every series, selects the primary
    via _select_primary_series, reuses the already-computed QC result for
    the selected series (no double work), and records selected=True/False
    in the per-series rows for the mixed_series_report.

    Returns
    -------
    qc_dict    — QC columns for the pipeline_report row
    mixed_rows — per-series dicts for mixed_series_report (empty if not mixed)
    """
    result = {col: None for col in _ALL_QC_COLUMNS}

    # 1. Load headers
    datasets, _ = load_headers(Path(dcm_path))
    if not datasets:
        result["corrupt_headers"] = True
        return result, []
    result["corrupt_headers"] = False

    # 2. Group by SeriesInstanceUID
    series_groups = group_by_series_uid(datasets)

    result["mixed_series"] = len(series_groups) > 1

    if result["mixed_series"]:
        # Characterise every series (needed both for selection and the report).
        series_summaries = []
        for uid, series_ds in series_groups.items():
            series_qc = _check_series(series_ds)
            series_summaries.append({"series_uid": str(uid), **series_qc})

        # Select the primary series using the documented algorithm.
        primary_uid = _select_primary_series(series_summaries)

        if primary_uid is None:
            # No axial 3D candidate — block the exam; still record all series.
            result["no_valid_series"] = True
            mixed_rows = []
            for s in series_summaries:
                s["selected"] = False
                mixed_rows.append({k: s.get(k) for k in _MIXED_SERIES_COLUMNS
                                   if k not in ("exam_id",)})
            return result, mixed_rows

        # Mark which series was selected and build mixed_series_report rows.
        mixed_rows = []
        primary_qc = None
        for s in series_summaries:
            s["selected"] = (s["series_uid"] == primary_uid)
            mixed_rows.append({k: s.get(k) for k in _MIXED_SERIES_COLUMNS
                               if k not in ("exam_id",)})
            if s["selected"]:
                primary_qc = s

        # Apply the selected series' QC to the exam-level result.
        for k, v in primary_qc.items():
            if k not in ("corrupt_headers", "mixed_series",
                         "series_uid", "selected"):
                result[k] = v
        result["selected_series_uid"] = primary_uid

    else:
        # Single series: run QC directly, no selection needed.
        series_result = _check_series(datasets)
        for k, v in series_result.items():
            if k not in ("corrupt_headers", "mixed_series"):
                result[k] = v
        mixed_rows = []

    return result, mixed_rows


def _compute_block_reason(row) -> str | None:
    """Derive qc_block_reason from all columns that can block conversion.

    Covers structural eligibility, metadata eligibility, and DICOM QC.
    Returns None if the exam is fully eligible, or a comma-separated string
    of blocking reasons. Consumers only need to check `qc_block_reason is None`.
    """
    reasons = []

    # Structural eligibility (folder / naming)
    if row.get("folder_status") != "ready":
        reasons.append("not_ready")
        return ",".join(reasons)  # nothing else is meaningful to check
    if row.get("name_parse_status") in ("unparseable", "gap"):
        reasons.append("unparseable")
        return ",".join(reasons)

    # Metadata eligibility — any gap in class/gender/age is one reason.
    if (pd.isna(row.get("class"))
            or pd.isna(row.get("gender"))
            or pd.isna(row.get("age"))):
        reasons.append("incomplete_metadata")
    elif row.get("class") in config.EXCLUDE_CLASSES:
        reasons.append("excluded_class")

    # DICOM QC
    if row.get("corrupt_headers"):          reasons.append("corrupt_headers")
    if row.get("missing_spatial_metadata"): reasons.append("missing_spatial_metadata")
    if row.get("slice_issue"):              reasons.append("slice_issue")
    if row.get("inconsistent_spacing"):     reasons.append("inconsistent_spacing")
    if row.get("extent_issue"):             reasons.append("extent_issue")
    if row.get("no_valid_series"):          reasons.append("no_valid_series")

    return ",".join(reasons) if reasons else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add QC columns to df. Returns (updated_df, mixed_series_df).

    mixed_series_df has one row per series inside mixed-series folders,
    with a `selected` column indicating which series was used for the
    pipeline_report row and subsequent conversion.

    It is an empty DataFrame (with correct columns) if no mixed-series
    exams were found.

    Pre-QC gate:
      folder_status == ready
      name_parse_status not in {unparseable, gap}
      class not in config.EXCLUDE_CLASSES
    """
    df = df.copy()
    for col in _ALL_QC_COLUMNS:
        df[col] = None

    eligible = (
        (df["folder_status"] == "ready") &
        (~df["name_parse_status"].isin(["unparseable", "gap"])) &
        (~df["class"].isin(config.EXCLUDE_CLASSES))
    )

    all_mixed_rows: list[dict] = []
    total = eligible.sum()

    for i, (idx, row) in enumerate(df[eligible].iterrows(), 1):
        if i % 50 == 0 or i == total:
            print(f"  QC: {i}/{total}", end="\r")
        qc, mixed_rows = _check_exam(row["dcm_path"])
        for col, val in qc.items():
            df.at[idx, col] = val
        for mrow in mixed_rows:
            all_mixed_rows.append({"exam_id": row["exam_id"], **mrow})

    print()

    df["qc_block_reason"] = df.apply(_compute_block_reason, axis=1)

    mixed_df = (
        pd.DataFrame(all_mixed_rows, columns=_MIXED_SERIES_COLUMNS)
        if all_mixed_rows
        else pd.DataFrame(columns=_MIXED_SERIES_COLUMNS)
    )
    return df, mixed_df
