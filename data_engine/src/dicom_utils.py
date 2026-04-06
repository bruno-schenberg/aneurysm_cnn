"""
dicom_utils.py

Validates raw DICOM series for integrity and geometric correctness, and produces
a breakdown of any folders that contain multiple mixed series.

## Why DICOM validation is necessary

Raw DICOM exports from hospital PACS systems are noisy. The same physical scanner
can produce files with:
- Corrupted or truncated headers
- Scout/localizer images mixed in with the diagnostic series
- Multiple series (e.g. contrast and non-contrast) dumped into the same folder
- Inconsistent or missing spatial metadata tags
- Non-consecutive or duplicated slice positions

A 3D CNN requires a clean, geometrically consistent stack of slices to form a valid
volumetric input. Any of the above problems, if undetected, would silently corrupt
the volume and poison the training data. This module detects and categorizes all
of these failure modes so they can be excluded before conversion.

## Slice sorting strategy

DICOM's InstanceNumber tag is meant to indicate slice order, but in practice it is
unreliable. PACS systems frequently reset or reorder it during export. Instead, this
module uses **projection-based sorting**: each slice's ImagePositionPatient (a 3D
coordinate in mm) is projected onto the scan's normal vector, which is derived from
the ImageOrientationPatient tag via cross product. The resulting scalar distance gives
a physically-correct, scanner-independent sort key.

## Validation outcomes

Each case receives one of the following validation_status values:

  OK                        — Series is geometrically clean and ready for conversion
  GAPPED_SEQUENCE           — Uniform spacing but non-consecutive InstanceNumbers;
                              some slices may be missing but physical geometry is intact
  GAPPED_SEQUENCE (N miss.) — 2× spacing gaps detected; N slices are missing
  VARIABLE_SPACING          — Inconsistent inter-slice distances; volume unreliable
  DUPLICATE_SLICES          — Two or more slices share the same physical position
  MIXED_SERIES_ERROR        — Multiple SeriesInstanceUIDs found in one folder
  NO_ORIGINAL_IMAGES_FOUND  — All images are scouts or derived; no diagnostic series
  NOT_3D_VOLUME_ERROR       — Only one slice found; cannot form a volume
  MISSING_SPATIAL_METADATA_ERROR — IOP, IPP, or PixelSpacing tags absent
  NO_DCM_FILES              — No .dcm files found at the resolved path
  ALL_DCMS_CORRUPT          — Every .dcm file failed to parse
  NOT_APPLICABLE            — Case was already excluded (EMPTY, MISSING, DUPLICATE_DATA)
  BELOW_LIMIT / ABOVE_LIMIT — Statistical outlier by physical exam size (IQR method)

Only OK cases are passed to the NIfTI conversion step. All other statuses
are excluded from conversion and recorded in the audit log only.

Public API (called by data_cleaner.py):
  - validate_dcms(data_mapping, base_path)  : main validation pipeline
  - analyze_mixed_folders(data_mapping, base_path, output_csv)  : mixed series report
"""

import logging
import os
import csv
from collections import defaultdict
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np

logger = logging.getLogger("dicom_ingestion")


# ----------------------------------------------------
# 1. Orientation Helpers
# ----------------------------------------------------


def get_orientation(iop: list[float] | None) -> str:
    """
    Classifies the scan plane from the ImageOrientationPatient (IOP) DICOM tag.

    Every DICOM file stores a tag called ImageOrientationPatient (IOP) that
    describes how the image is oriented relative to the patient's body. It is a
    list of 6 numbers that encode the directions of the image's rows and columns
    in 3D space. By inspecting these numbers we can tell whether the scanner was
    positioned to acquire an axial (top-down), coronal (front-back), or sagittal
    (side-to-side) slice.

    Real scanners are never perfectly aligned — a patient lying slightly tilted
    will produce values like [0.9998, 0.0017, 0, 0, 1, 0] instead of the ideal
    [1, 0, 0, 0, 1, 0]. Rounding each value to the nearest whole number corrects
    for these small physical imperfections so the orientation is still recognised
    correctly, without misclassifying a genuinely oblique acquisition.

    Standard orientations and their rounded IOP signatures:
      AXIAL    — [1, 0, 0, 0, 1, 0] : horizontal cross-section through the brain
      CORONAL  — [1, 0, 0, 0, 0, -1]: front-to-back cross-section
      SAGITTAL — [0, 1, 0, 0, 0, -1]: left-to-right cross-section

    Any other combination is classified as OBLIQUE. Oblique scans are not
    automatically rejected — they are recorded in the audit log and can be
    reviewed manually.

    Args:
        iop: The ImageOrientationPatient tag value as a list of 6 floats,
             or None if the tag is absent.

    Returns:
        One of: ``'AXIAL'``, ``'CORONAL'``, ``'SAGITTAL'``, ``'OBLIQUE'``, ``'UNKNOWN'``.
    """
    if iop is None or len(iop) < 6:
        return "UNKNOWN"

    # Round to nearest integer to handle slight scanner tilts.
    r = [round(x) for x in iop]

    if r == [1, 0, 0, 0, 1, 0]:
        return "AXIAL"
    elif r == [1, 0, 0, 0, 0, -1]:
        return "CORONAL"
    elif r == [0, 1, 0, 0, 0, -1]:
        return "SAGITTAL"
    else:
        return "OBLIQUE"


# ----------------------------------------------------
# 2. DICOM Loading
# ----------------------------------------------------


def load_dicom_metadata(path: str) -> list:
    """
    Reads the DICOM header of every ``.dcm`` file in ``path`` and returns them
    as a list of pydicom Dataset objects.

    ``stop_before_pixels=True`` tells pydicom to parse only the file metadata
    and tag headers, stopping before it reads the pixel data array. This makes
    the load roughly 10–100× faster per file and uses a fraction of the memory,
    which matters when scanning thousands of slices just to check geometry.
    Pixel data is only needed at conversion time (``nifti_utils.py``), not here.

    Invalid or corrupted DICOM files are skipped with a warning rather than
    crashing the pipeline. A corrupted header is not necessarily fatal — the
    rest of the series may still be valid. If all files in a folder are corrupt,
    ``validate_dcms`` catches the empty list and assigns ``ALL_DCMS_CORRUPT``.

    Args:
        path: Absolute path to the directory containing ``.dcm`` files.

    Returns:
        List of pydicom Dataset objects, one per successfully parsed file.
        Files that fail to parse are omitted from the list.
    """
    metadata_list = []
    for f in os.listdir(path):
        if f.lower().endswith('.dcm'):
            try:
                dcm_path = os.path.join(path, f)
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                metadata_list.append(ds)
            except InvalidDicomError:
                logger.warning(f"  - Warning: Skipping corrupt/invalid DICOM file: {dcm_path}")
    return metadata_list


# ----------------------------------------------------
# 3. Slice Sorting and Spacing Analysis (Private)
# ----------------------------------------------------


def _sort_slices_by_projection(metadata_list: list, iop: list[float]) -> None:
    """
    Sorts DICOM slices into correct anatomical order using projection-based sorting.

    **Why not use InstanceNumber?**
    Each DICOM file contains an InstanceNumber tag that is supposed to indicate
    slice order. In practice, PACS systems frequently reset or scramble it during
    export, so it cannot be trusted. Instead, every DICOM file also stores the
    real-world 3D coordinates of where that slice was physically acquired
    (ImagePositionPatient, a point in millimetres). This function uses those
    coordinates to determine the correct anatomical order — independently of
    whatever InstanceNumber says.

    **How it works (in plain terms):**
    Think of the scan as a stack of parallel photograph slices through the brain.
    Each slice knows exactly where it sits in 3D space (its coordinates in mm).
    To sort the stack, we need a single number per slice that represents "how far
    along the stack" it is. We get that number by measuring each slice's position
    along the axis that runs perpendicular to the imaging plane — the stacking
    direction. The IOP tag tells us which direction that is. Once every slice has
    that single distance value, sorting by it gives the correct anatomical order.

    Each Dataset in ``metadata_list`` receives a ``.dist`` attribute (float, mm)
    after this call. This attribute is consumed by ``_evaluate_spacing`` to
    compute inter-slice gaps.

    Args:
        metadata_list: List of pydicom Datasets for a single series.
                       Modified in-place: sorted and ``.dist`` added.
        iop: ImageOrientationPatient tag value as a list of 6 floats.
    """
    row_vec = np.array(iop[:3])
    col_vec = np.array(iop[3:])
    # Compute the stacking direction: the axis perpendicular to the imaging plane.
    normal_vec = np.cross(row_vec, col_vec)

    for ds in metadata_list:
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp is None:
            ds.dist = 0  # Failsafe; MISSING_SPATIAL_METADATA_ERROR would have caught this earlier
        else:
            # Distance of this slice along the stacking axis, in mm.
            ds.dist = np.dot(np.array(ipp), normal_vec)

    metadata_list.sort(key=lambda x: x.dist)


def _evaluate_spacing(metadata_list: list) -> tuple:
    """
    Analyses the inter-slice distances of a sorted series to assess its geometric
    integrity and assign a validation status.

    Relies on the ``.dist`` attribute set by ``_sort_slices_by_projection``.
    Computes the absolute difference between consecutive distances (deltas), rounds
    to 2 decimal places (scanner floating-point precision), and classifies the
    result:

    **DUPLICATE_SLICES** — One or more deltas < 0.001 mm. Two slices occupy the
      same physical position. This often indicates a PACS export error or that the
      same series was exported twice. The volume cannot be used as-is.

    **OK** — Exactly one unique spacing (perfectly uniform stack) AND consecutive
      InstanceNumbers. This is the ideal case.

    **GAPPED_SEQUENCE** — Exactly one unique spacing (uniform physical gaps) but
      non-consecutive InstanceNumbers. The geometry is intact, but the instance
      numbering suggests some slices may have been dropped. Included in conversion
      with a warning.

    **GAPPED_SEQUENCE (N missing)** — Two unique spacings where one is
      approximately 2× the other (within 0.1 mm tolerance). This pattern means N
      specific gaps where exactly one slice is absent. The number of such gaps is
      reported.

    **VARIABLE_SPACING** — Any other multi-spacing case. The inter-slice geometry
      is inconsistent; the volume cannot be reliably interpreted.

    Args:
        metadata_list: List of pydicom Datasets sorted by ``.dist``.

    Returns:
        Tuple of ``(validation_status, unique_spacings, duplicate_count, deltas)``:
          - ``validation_status``: string classification (see above)
          - ``unique_spacings``: numpy array of unique rounded delta values (mm)
          - ``duplicate_count``: number of zero-distance pairs (0 if no duplicates)
          - ``deltas``: numpy array of all raw absolute inter-slice distances (mm)
    """
    distances = np.array([ds.dist for ds in metadata_list])
    deltas = np.abs(np.diff(distances))
    unique_spacings = np.unique(np.around(deltas, decimals=2))

    # Count slice pairs that are at the same physical position (delta < 0.001 mm).
    num_duplicates = int(np.sum(deltas < 0.001))

    if num_duplicates > 0:
        return 'DUPLICATE_SLICES', unique_spacings, num_duplicates, deltas

    if len(unique_spacings) == 1:
        # Physical spacing is perfectly uniform. Now check InstanceNumbers:
        # a gap in instance numbers at uniform physical spacing suggests that
        # some slices were dropped but those that remain are geometrically sound.
        instance_numbers = sorted([int(getattr(ds, 'InstanceNumber', 0)) for ds in metadata_list])
        is_gapped_instance = any(np.diff(instance_numbers) != 1)
        return ('GAPPED_SEQUENCE' if is_gapped_instance else 'OK'), unique_spacings, 0, deltas

    elif len(unique_spacings) == 2:
        # Two distinct spacings found. Check whether one is approximately twice
        # the other — the hallmark of a single missing slice between some pairs.
        sorted_spacings = sorted(unique_spacings)
        if np.isclose(sorted_spacings[1], 2 * sorted_spacings[0], atol=0.1):
            gaps = np.sum(np.isclose(deltas, sorted_spacings[1], atol=0.1))
            return f'GAPPED_SEQUENCE ({gaps} missing)', unique_spacings, 0, deltas
        else:
            # Two unrelated spacings — geometry is genuinely inconsistent.
            return 'VARIABLE_SPACING', unique_spacings, 0, deltas

    else:
        # Three or more distinct spacings — too irregular to trust.
        return 'VARIABLE_SPACING', unique_spacings, 0, deltas


# ----------------------------------------------------
# 4. Main Series Validation
# ----------------------------------------------------


def validate_dcms(data_mapping: list[dict], base_path: str) -> list[dict]:
    """
    Runs the full DICOM series validation pipeline on every case in ``data_mapping``.

    Iterates through each case record and performs a sequence of checks, stopping
    at the first failure and recording the reason. Cases that pass all checks
    receive a spacing-based status from ``_evaluate_spacing`` (``OK``,
    ``GAPPED_SEQUENCE``, etc.). After all cases are checked, a statistical
    outlier pass flags cases with abnormally small or large physical coverage.

    Validation checks (in order):

      1. **Skip non-processable cases** — cases with ``data_path`` in
         ``{EMPTY, MISSING, DUPLICATE_DATA}`` are marked ``NOT_APPLICABLE``
         and skipped immediately.

      2. **DCM file existence** — if the resolved path contains no ``.dcm`` files,
         mark ``NO_DCM_FILES``. This catches cases where ``file_utils`` resolved a
         path that exists on disk but contains no DICOM data.

      3. **Header loading** — if all files fail to parse, mark ``ALL_DCMS_CORRUPT``.

      4. **Mixed series detection** — a folder should contain exactly one
         SeriesInstanceUID. Multiple UIDs indicate that two or more acquisitions
         (e.g. contrast and non-contrast) were exported into the same folder.
         These are marked ``MIXED_SERIES_ERROR`` and reported separately by
         ``analyze_mixed_folders`` for manual review.

      5. **Scout image filtering** — DICOM files with ImageType not containing
         ``'ORIGINAL'`` are derived images (scouts, localizers, reformats). They
         are excluded from the slice count; only ORIGINAL slices are validated
         and converted.

      6. **3D volume check** — a series with only one slice cannot form a volume.
         Marked ``NOT_3D_VOLUME_ERROR``.

      7. **Spatial metadata check** — ImageOrientationPatient, ImagePositionPatient,
         and PixelSpacing are all required for projection-based sorting and correct
         NIfTI geometry. If any are absent, marked
         ``MISSING_SPATIAL_METADATA_ERROR``.

      8. **Projection-based sorting** — slices are sorted by physical position
         (see ``_sort_slices_by_projection``).

      9. **Spacing analysis** — inter-slice distances are evaluated and a status
         is assigned (see ``_evaluate_spacing``).

     10. **Metadata extraction** — orientation, modality, patient sex, image
         dimensions, effective slice thickness, and physical exam size (slice count
         × thickness, in mm) are recorded for the audit log.

         Note: PatientAge is not extracted from the DICOM headers here. It is
         instead read from the ``Age`` column in ``classes.csv`` by
         ``class_utils.join_class_data``, which runs after this step.

     11. **Outlier detection** — after all cases are processed,
         ``_flag_exam_size_outliers`` flags cases whose physical coverage falls
         outside the IQR fences of the OK-case distribution.

    Args:
        data_mapping: List of case record dicts as produced by ``file_utils.organize_data``.
                      Each dict is updated in-place with validation results.
        base_path: Absolute path to the raw DICOM root directory.

    Returns:
        The same list with validation results merged into every record.
    """
    logger.info("\nValidating DICOM series with Projection-Based Sorting...")

    for item in data_mapping:
        data_path = item.get('data_path')

        # 1. Skip cases that were already excluded by file_utils (no valid DICOM path).
        if data_path in ('EMPTY', 'MISSING', 'DUPLICATE_DATA'):
            item['validation_status'] = 'NOT_APPLICABLE'
            continue

        logger.info(f"  - Validating {item['fixed_name']}...")
        full_path = os.path.join(base_path, data_path)

        try:
            # 2. Verify that .dcm files exist at the resolved path before loading.
            dcm_files = [f for f in os.listdir(full_path) if f.lower().endswith('.dcm')]
            if not dcm_files:
                item['validation_status'] = 'NO_DCM_FILES'
                continue

            # 3. Load all DICOM headers (pixel data is not needed at this stage).
            metadata_list = load_dicom_metadata(full_path)

            if not metadata_list:
                item['validation_status'] = 'ALL_DCMS_CORRUPT'
                continue

            # 4. Detect mixed series: each unique SeriesInstanceUID represents a
            #    separate acquisition. A folder should contain exactly one.
            uids = {ds.SeriesInstanceUID for ds in metadata_list}
            if len(uids) > 1:
                item['validation_status'] = f'MIXED_SERIES_ERROR ({len(uids)} UIDs)'
                continue

            # 5. Filter out scout/localizer images.
            #    ORIGINAL images are the primary diagnostic series (the actual scan).
            #    Non-ORIGINAL images (DERIVED, SECONDARY) are scouts, reformats, or
            #    post-processed images that should not be stacked into the volume.
            main_series_metadata = [ds for ds in metadata_list if 'ORIGINAL' in getattr(ds, 'ImageType', [])]

            if not main_series_metadata:
                item['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
                continue

            # Update total_dcms to reflect the number of usable slices after filtering.
            item['total_dcms'] = len(main_series_metadata)
            scout_count = len(metadata_list) - len(main_series_metadata)
            item['scout_slice_count'] = scout_count
            if scout_count > 0:
                logger.warning(f"  - {item['fixed_name']}: filtered out {scout_count} scout/derived slice(s)")

            # 6. Reject single-slice series: cannot form a 3D volume for the CNN.
            if len(main_series_metadata) == 1:
                item['validation_status'] = 'NOT_3D_VOLUME_ERROR'
                logger.warning(f"  - {item['fixed_name']}: only 1 slice found, cannot form a 3D volume")
                continue

            # 7. Check that the mandatory spatial metadata tags are present.
            #    IOP and IPP are required for projection-based sorting.
            #    PixelSpacing is required for correct NIfTI affine construction.
            sample_ds = main_series_metadata[0]
            iop = getattr(sample_ds, 'ImageOrientationPatient', None)
            ipp = getattr(sample_ds, 'ImagePositionPatient', None)

            if iop is None or ipp is None or getattr(sample_ds, 'PixelSpacing', None) is None:
                item['validation_status'] = 'MISSING_SPATIAL_METADATA_ERROR'
                continue

            # 8. Sort slices into anatomical order using physical coordinates.
            _sort_slices_by_projection(main_series_metadata, iop)

            # 9. Evaluate inter-slice spacing to classify geometric integrity.
            status, unique_spacings, duplicates, deltas = _evaluate_spacing(main_series_metadata)
            item['validation_status'] = status
            item['duplicate_slice_count'] = duplicates

            # 10. Extract summary metadata for the audit log.
            rows = getattr(sample_ds, 'Rows', 'N/A')
            cols = getattr(sample_ds, 'Columns', 'N/A')

            # Effective slice thickness: use the most common inter-slice delta rather
            # than the SliceThickness DICOM tag, which can differ from the actual
            # reconstructed spacing (e.g. when overlapping reconstructions are used).
            if len(deltas) > 0:
                vals, counts = np.unique(np.around(deltas, decimals=2), return_counts=True)
                effective_slice_thickness = float(vals[np.argmax(counts)])
            else:
                effective_slice_thickness = 'N/A'

            item['orientation'] = get_orientation(iop)
            item['modality'] = getattr(sample_ds, 'Modality', 'N/A')
            item['patient_sex'] = getattr(sample_ds, 'PatientSex', 'UNKNOWN')
            item['image_dimensions'] = f"{rows}x{cols}"

            if isinstance(effective_slice_thickness, float):
                item['slice_thickness'] = round(effective_slice_thickness, 2)
                # Physical coverage in mm: used by _flag_exam_size_outliers to
                # detect scans that are implausibly short or long.
                item['exam_size'] = round(item['total_dcms'] * effective_slice_thickness, 2)
            else:
                item['slice_thickness'] = 'N/A'
                item['exam_size'] = 'N/A'

        except Exception as e:
            item['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

    # 11. Statistical outlier detection on physical exam size across all OK cases.
    data_mapping = _flag_exam_size_outliers(data_mapping)

    logger.info("Validation complete.")
    return data_mapping


def _flag_exam_size_outliers(data_mapping: list[dict]) -> list[dict]:
    """
    Flags validated cases whose physical exam coverage is a statistical outlier.

    **Rationale:** A healthy brain CTA/MRA typically covers a consistent anatomical
    range (roughly 100–200 mm of axial coverage). Cases far outside this range are
    likely partial scans, planning scans, or mis-exported series. Including them in
    training would add noise — the CNN would see volumes with very different fields
    of view labeled as the same class.

    **Method — Tukey's IQR fences:**
    The IQR (interquartile range) method is used rather than a z-score because it
    is robust to non-normal distributions and to the small sample sizes typical of
    medical imaging datasets. The fences are:

      lower_fence = Q1 − 1.5 × IQR
      upper_fence = Q3 + 1.5 × IQR

    Cases below the lower fence are marked ``BELOW_LIMIT``.
    Cases above the upper fence are marked ``ABOVE_LIMIT``.

    Only cases with ``validation_status == 'OK'`` are considered — cases that
    already failed validation are not re-classified here. The outlier check is
    run after all individual validations are complete so that the fence values
    are computed from the clean, validated subset only.

    Args:
        data_mapping: Full list of case record dicts from ``validate_dcms``.

    Returns:
        The same list with ``validation_status`` updated to ``BELOW_LIMIT`` or
        ``ABOVE_LIMIT`` for outlier cases (in-place).
    """
    ok_exams = [item for item in data_mapping if item.get('validation_status') == 'OK']

    exam_sizes = []
    for item in ok_exams:
        try:
            exam_sizes.append(float(item['exam_size']))
        except (ValueError, TypeError):
            continue

    if not exam_sizes:
        return data_mapping

    q1 = np.percentile(exam_sizes, 25)
    q3 = np.percentile(exam_sizes, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    logger.info("\nAnalyzing exam size distribution for 'OK' series...")
    logger.info(f"  - Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    logger.info(f"  - Lower Fence: {lower_fence:.2f}, Upper Fence: {upper_fence:.2f}")

    for item in ok_exams:
        if isinstance(item.get('exam_size'), (int, float, np.number)):
            if item['exam_size'] < lower_fence:
                item['validation_status'] = 'BELOW_LIMIT'
            elif item['exam_size'] > upper_fence:
                item['validation_status'] = 'ABOVE_LIMIT'

    outliers_below = [item for item in ok_exams if item.get('validation_status') == 'BELOW_LIMIT']
    outliers_above = [item for item in ok_exams if item.get('validation_status') == 'ABOVE_LIMIT']

    if outliers_below:
        logger.info(f"\nFound {len(outliers_below)} exams BELOW the lower fence:")
        for item in outliers_below:
            logger.info(f"  - {item['fixed_name']} (Exam Size: {item['exam_size']:.2f})")

    if outliers_above:
        logger.info(f"\nFound {len(outliers_above)} exams ABOVE the upper fence:")
        for item in outliers_above:
            logger.info(f"  - {item['fixed_name']} (Exam Size: {item['exam_size']:.2f})")

    return data_mapping


# ----------------------------------------------------
# 5. Mixed Series Analysis
# ----------------------------------------------------


def analyze_mixed_series(folder_path: str) -> list[dict]:
    """
    Produces a per-series validation report for a folder flagged as MIXED_SERIES_ERROR.

    When ``validate_dcms`` finds multiple SeriesInstanceUIDs in one folder, it cannot
    automatically determine which series to use — this requires human judgment (e.g.
    choosing the axial diagnostic series over a coronal reformatted series). This
    function provides the information needed to make that decision by running a
    mini-validation on each series individually.

    The function groups all DICOM files in ``folder_path`` by their SeriesInstanceUID
    and, for each group:
    - Filters out scout/localizer images (non-ORIGINAL).
    - Extracts orientation, modality, slice thickness, image dimensions, and patient
      demographics.
    - Runs projection-based sorting and spacing analysis (the same checks as the
      main validation pipeline) so the reviewer can see whether each individual
      series is geometrically sound.

    The resulting report is written to ``mixed_series_analysis.csv`` by
    ``analyze_mixed_folders`` for manual review.

    Args:
        folder_path: Absolute path to the case folder containing mixed series.

    Returns:
        List of per-series report dicts. Empty list if the folder does not exist
        or contains no readable DICOM files.
    """
    if not os.path.isdir(folder_path):
        return []

    all_metadata = load_dicom_metadata(folder_path)
    if not all_metadata:
        return []

    # Group files by SeriesInstanceUID — each UID is one acquisition.
    series_groups = defaultdict(list)
    for ds in all_metadata:
        series_groups[ds.SeriesInstanceUID].append(ds)

    series_reports = []
    for i, (uid, series_metadata) in enumerate(series_groups.items()):
        report = {
            'series_index': i + 1,
            'series_instance_uid': uid,
            'total_dcms': len(series_metadata),
            'validation_status': 'PENDING',
        }

        # Apply the same scout-filtering logic as the main validation pipeline.
        main_series_metadata = [ds for ds in series_metadata if 'ORIGINAL' in getattr(ds, 'ImageType', [])]
        report['scout_slice_count'] = len(series_metadata) - len(main_series_metadata)
        report['total_dcms'] = len(main_series_metadata)

        if not main_series_metadata:
            report['validation_status'] = 'NO_ORIGINAL_IMAGES_FOUND'
            series_reports.append(report)
            continue

        sample_ds = main_series_metadata[0]
        iop = getattr(sample_ds, 'ImageOrientationPatient', None)

        report['orientation'] = get_orientation(iop)
        report['modality'] = getattr(sample_ds, 'Modality', 'N/A')
        report['slice_thickness'] = getattr(sample_ds, 'SliceThickness', 'N/A')
        report['patient_age'] = getattr(sample_ds, 'PatientAge', 'N/A')
        report['patient_sex'] = getattr(sample_ds, 'PatientSex', 'N/A')
        report['exam_date'] = getattr(sample_ds, 'StudyDate', 'N/A')
        report['image_dimensions'] = f"{getattr(sample_ds, 'Rows', 'N/A')}x{getattr(sample_ds, 'Columns', 'N/A')}"

        try:
            if iop is not None and getattr(sample_ds, 'ImagePositionPatient', None) is not None:
                _sort_slices_by_projection(main_series_metadata, iop)
                status, _, duplicates, _ = _evaluate_spacing(main_series_metadata)
                report['validation_status'] = status
                report['duplicate_slice_count'] = duplicates
            else:
                report['validation_status'] = 'MISSING_MANDATORY_SPATIAL_TAGS'

        except Exception as e:
            report['validation_status'] = f'VALIDATION_ERROR: {str(e)}'

        series_reports.append(report)

    return series_reports


def analyze_mixed_folders(data_mapping: list[dict], base_path: str, output_csv_path: str) -> None:
    """
    Collects all MIXED_SERIES_ERROR cases and writes a per-series breakdown CSV.

    Iterates over all cases in ``data_mapping`` that were flagged as
    ``MIXED_SERIES_ERROR`` by ``validate_dcms``, calls ``analyze_mixed_series``
    on each, and writes the combined report to ``output_csv_path``.

    The CSV is intended for manual review: a researcher reads it to decide which
    series within each mixed folder is the correct diagnostic series, then manually
    moves or renames the files so a re-run of ``data_cleaner.py`` can process them
    as ``READY`` or ``SUBFOLDER_PATH`` cases.

    If no mixed cases are found, the function exits immediately without creating
    the CSV. The ``parent_case_name`` column is placed first in the CSV so the
    reader can immediately identify which patient each series row belongs to.

    Args:
        data_mapping: Full list of case record dicts from ``validate_dcms``.
        base_path: Absolute path to the raw DICOM root directory.
        output_csv_path: Absolute path where the mixed series CSV will be written.
    """
    mixed_cases = [item for item in data_mapping if str(item.get('validation_status')).startswith('MIXED_SERIES_ERROR')]
    if not mixed_cases:
        return

    logger.info(f"Analyzing {len(mixed_cases)} cases with mixed series...")
    all_series_reports = []

    for case in mixed_cases:
        folder_path = os.path.join(base_path, case['data_path'])
        reports = analyze_mixed_series(folder_path)
        for report in reports:
            report['parent_case_name'] = case['fixed_name']
        all_series_reports.extend(reports)

    if all_series_reports:
        # Collect all keys that appear across any report, then sort for
        # consistent column ordering, with parent_case_name pinned first.
        all_keys = set()
        for report in all_series_reports:
            all_keys.update(report.keys())

        fieldnames = sorted(list(all_keys))
        if 'parent_case_name' in fieldnames:
            fieldnames.remove('parent_case_name')
            fieldnames.insert(0, 'parent_case_name')

        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_series_reports)
        logger.info(f"Mixed series analysis complete. Saved to: {output_csv_path}")
