"""
Batch Extraction & Inventory Scan Pipeline.

Extracts all zip archives found in a source directory into a single output
directory, then runs the full data-engine pipeline (organisation, DICOM
validation, classification join, missing-class detection, and gender check)
without performing any NIfTI conversion.

Useful for auditing a new batch of raw data: quickly identify which cases are
present, which are missing, which have DICOM issues, and which are missing
clinical labels or gender data — before committing to a lengthy conversion run.

Usage:
    # Full mode: extract all zips at once, then scan
    python inventory_scan.py

    # Partitioned mode: extract one zip at a time, scan, delete, repeat
    # Use this when local disk space is limited.
    python inventory_scan.py --partitioned

    # Skip extraction if already extracted, just scan
    python inventory_scan.py --skip-extract
"""

import argparse
import csv
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from src.file_utils import get_subfolders, organize_data, find_missing_cases
from src.dicom_utils import validate_dcms, analyze_mixed_folders
from src.class_utils import join_class_data
from src.logging_utils import setup_logger


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "inventory"

DEFAULT_ZIP_DIR = Path("/mnt/data/raw")
DEFAULT_EXTRACT_DIR = Path("/mnt/data/raw_extract")
CLASSES_CSV_PATH = DATASET_DIR / "classes.csv"

SEX_TO_GENDER = {"M": 0, "F": 1}

SUMMARY_FIELDNAMES = [
    "original_name", "fixed_name", "data_path", "total_dcms",
    "validation_status", "class_status", "gender_status",
    "duplicate_slice_count", "scout_slice_count",
    "orientation", "modality", "slice_thickness", "patient_age",
    "patient_sex", "image_dimensions", "class", "location",
    "exam_size",
]


# ---------------------------------------------------------------------------
# Class status (independent of validation_status)
# ---------------------------------------------------------------------------


def add_class_status(data: list[dict], classes_csv: Path, logger) -> list[dict]:
    """
    Adds a ``class_status`` field to every case record, independently of
    ``validation_status``.

    Unlike ``check_missing_class`` (which only examines OK cases and writes
    its finding into ``validation_status``), this function checks every case —
    including those that are filesystem-MISSING or otherwise not processable —
    so that a case which is both absent from the download AND unlabelled is
    visible as ``MISSING`` in both columns simultaneously.

    Values:
      ``OK``      — a matching entry exists in classes.csv
      ``MISSING`` — no entry found (case is unlabelled)

    Duplicate-suffixed names (e.g. ``BP001A``) are stripped to their base name
    (``BP001``) before the CSV lookup, consistent with ``class_utils`` behaviour.

    Args:
        data: Full list of case record dicts (including MISSING sentinel rows).
        classes_csv: Path to ``dataset/classes.csv``.
        logger: Logger instance.

    Returns:
        The same list with ``'class_status'`` added to every record in-place.
    """
    logger.info("\nChecking class label status for all cases...")

    labelled: set[str] = set()
    if classes_csv.exists():
        with open(classes_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("exam", "").strip().upper()
                class_val = row.get("class", "").strip()
                # Only count as labelled if the class field is actually filled in.
                if name and class_val:
                    labelled.add(name)
    else:
        logger.warning(f"  - classes.csv not found at '{classes_csv}'. All cases will be MISSING.")

    missing_count = 0
    for item in data:
        fixed_name = item.get("fixed_name", "")
        m = re.match(r"(BP\d+)", fixed_name)
        base_name = m.group(1) if m else fixed_name.upper()

        if base_name in labelled:
            item["class_status"] = "OK"
        else:
            item["class_status"] = "MISSING"
            missing_count += 1

    logger.info(f"  - {missing_count} case(s) have no class label in classes.csv.")
    return data


# ---------------------------------------------------------------------------
# Gender helpers
# ---------------------------------------------------------------------------


def check_missing_gender(data: list[dict], logger) -> list[dict]:
    """
    Adds a ``gender_status`` field to every case record, independently of
    ``validation_status``.

    Every case is checked: ``'OK'`` if ``patient_sex`` contains 'M' or 'F',
    ``'MISSING'`` for everything else. No exceptions based on validation
    outcome — a case that failed DICOM validation but whose sex tag was
    still readable will show ``'OK'``, and a case that was never downloaded
    will correctly show ``'MISSING'``.

    Args:
        data: List of case record dicts after the full pipeline has run.
        logger: Logger instance.

    Returns:
        The same list with ``'gender_status'`` added to every record in-place.
    """
    logger.info("\nChecking for missing gender (PatientSex) in all exams...")
    missing_count = 0

    for item in data:
        sex = str(item.get("patient_sex", "")).strip().upper()
        if sex in SEX_TO_GENDER:
            item["gender_status"] = "OK"
        else:
            item["gender_status"] = "MISSING"
            missing_count += 1

    logger.info(f"  - Flagged {missing_count} exam(s) with missing gender.")
    return data


def write_gender_to_classes_csv(data: list[dict], classes_csv: Path, logger) -> None:
    """
    Writes PatientSex data extracted from DICOM headers back to ``classes.csv``
    as a ``gender`` column (M → 0, F → 1).

    Mirrors the write-back logic of ``diagnostics/extract_gender.py``. Only
    writes new values — existing non-empty gender entries in the CSV are never
    overwritten. Cases whose ``patient_sex`` is not 'M' or 'F' are skipped.
    Duplicate-suffixed names (e.g. ``BP001A``) are stripped to their base name
    (``BP001``) before the CSV lookup, consistent with ``class_utils`` behaviour.

    Args:
        data: Full list of case record dicts after pipeline and gender check.
        classes_csv: Path to ``dataset/classes.csv``.
        logger: Logger instance.
    """
    if not classes_csv.exists():
        logger.warning(f"  - classes.csv not found at '{classes_csv}'. Skipping gender write-back.")
        return

    with open(classes_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames: list[str] = list(reader.fieldnames or [])
        rows = list(reader)

    row_by_exam = {row["exam"].strip().upper(): i for i, row in enumerate(rows)}

    if "gender" not in fieldnames:
        fieldnames.append("gender")

    written = 0
    for item in data:
        sex = str(item.get("patient_sex", "")).strip().upper()
        if sex not in SEX_TO_GENDER:
            continue

        # Strip disambiguating suffix (e.g. BP001A → BP001) for CSV lookup.
        fixed_name = item.get("fixed_name", "")
        m = re.match(r"(BP\d+)", fixed_name)
        base_name = m.group(1) if m else fixed_name.upper()

        idx = row_by_exam.get(base_name)
        if idx is None:
            continue

        existing = str(rows[idx].get("gender", "")).strip()
        if not existing:
            rows[idx]["gender"] = SEX_TO_GENDER[sex]
            written += 1

    with open(classes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  - Gender write-back: {written} new value(s) written to '{classes_csv}'.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _find_zips(zip_dir: Path, logger) -> list[Path]:
    """Returns sorted list of .zip files in zip_dir, logging a warning if none found."""
    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No .zip files found in '{zip_dir}'.")
    else:
        logger.info(f"Found {len(zip_files)} zip archive(s) in '{zip_dir}'.")
    return zip_files


def _extract_zip(zip_path: Path, extract_dir: Path, logger) -> bool:
    """
    Extracts a single zip archive into extract_dir.

    Returns True on success, False if the archive is invalid or extraction fails.
    """
    logger.info(f"  Extracting: {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info(f"    Done: {zip_path.name}")
        return True
    except zipfile.BadZipFile as e:
        logger.error(f"    Skipping {zip_path.name}: not a valid zip archive ({e})")
        return False
    except Exception as e:
        logger.error(f"    Failed to extract {zip_path.name}: {e}")
        return False


def _run_pipeline_steps(extract_dir: Path, mixed_csv_path: Path, logger) -> list[dict]:
    """
    Runs organisation, DICOM validation, mixed-series analysis, and class
    joining on a directory of extracted cases.

    Does NOT include MISSING sentinel rows — those are injected by
    organize_data internally but are meaningless when processing a partial
    batch. They are filtered out here and recomputed at the end over the
    full merged dataset.

    Args:
        extract_dir: Directory containing the extracted case folders.
        mixed_csv_path: File path for the mixed-series analysis CSV output.
        logger: Logger instance.

    Returns:
        List of case record dicts, with MISSING rows removed.
    """
    extract_dir_str = str(extract_dir)
    case_folders = get_subfolders(extract_dir_str)
    logger.info(f"  Found {len(case_folders)} case folder(s).")

    organized_data = organize_data(case_folders, extract_dir_str)
    validated_data = validate_dcms(organized_data, extract_dir_str)
    analyze_mixed_folders(validated_data, extract_dir_str, str(mixed_csv_path))
    data_with_classes = join_class_data(validated_data, str(CLASSES_CSV_PATH))

    # Strip MISSING sentinel rows — these are only meaningful across the full
    # dataset, not within a single partial batch.
    # Note: check_missing_class is intentionally NOT called here — class label
    # presence is tracked independently in the class_status column by
    # add_class_status, which runs once over the full merged dataset.
    return [row for row in data_with_classes if row.get("data_code") != "MISSING"]


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merge_mixed_csvs(csv_paths: list[Path], output_path: Path) -> None:
    """Concatenates multiple mixed-series CSVs into one, keeping a single header."""
    rows = []
    fieldnames = None
    for p in csv_paths:
        if not p.exists():
            continue
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)
    if not rows or fieldnames is None:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log_status_breakdown(data: list[dict], logger) -> None:
    status_counts = Counter(row.get("validation_status", "UNKNOWN") for row in data)
    class_missing  = sum(1 for row in data if row.get("class_status") == "MISSING")
    gender_missing = sum(1 for row in data if row.get("gender_status") == "MISSING")

    logger.info("\n--- Validation Status Breakdown ---")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status:<35} {count}")
    logger.info(f"  {'TOTAL':<35} {len(data)}")
    logger.info("\n--- Labels & Metadata ---")
    logger.info(f"  {'Missing class label (any status)':<35} {class_missing}")
    logger.info(f"  {'Missing PatientSex':<35} {gender_missing}")


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------


def run_inventory(
    zip_dir: Path,
    extract_dir: Path,
    skip_extract: bool = False,
    log_dir: Path | None = None,
) -> None:
    """
    Extracts all zips at once and runs the full pipeline in a single pass.

    Use this when disk space allows holding all extracted cases simultaneously.
    """
    out_dir = log_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out_dir / "inventory.log"))
    logger.info("=== Batch Extraction & Inventory Scan (full mode) ===")

    if skip_extract:
        logger.info("Skipping extraction step (--skip-extract).")
    else:
        if not zip_dir.exists():
            raise RuntimeError(f"Zip source directory not found: '{zip_dir}'")
        extract_dir.mkdir(parents=True, exist_ok=True)
        for zip_path in _find_zips(zip_dir, logger):
            _extract_zip(zip_path, extract_dir, logger)

    if not extract_dir.exists():
        raise RuntimeError(
            f"Extract directory does not exist: '{extract_dir}'. "
            "Run without --skip-extract to populate it first."
        )

    logger.info(f"\nScanning: '{extract_dir}'")
    final_data = _run_pipeline_steps(
        extract_dir,
        mixed_csv_path=out_dir / "mixed_series_analysis.csv",
        logger=logger,
    )

    # Reattach real MISSING rows across the full dataset and join class data
    # so their class/location/patient_age fields are populated from classes.csv.
    missing_rows = find_missing_cases(final_data)
    for row in missing_rows:
        row["validation_status"] = "MISSING"
    missing_rows = join_class_data(missing_rows, str(CLASSES_CSV_PATH))
    final_data.extend(missing_rows)

    add_class_status(final_data, CLASSES_CSV_PATH, logger)
    check_missing_gender(final_data, logger)
    write_gender_to_classes_csv(final_data, CLASSES_CSV_PATH, logger)

    _write_summary_csv(final_data, out_dir / "inventory_summary.csv")
    logger.info(f"\nSummary CSV written to: '{out_dir / 'inventory_summary.csv'}'")
    _log_status_breakdown(final_data, logger)
    logger.info("\nInventory scan complete.")


# ---------------------------------------------------------------------------
# Partitioned mode
# ---------------------------------------------------------------------------


def run_partitioned(
    zip_dir: Path,
    extract_dir: Path,
    log_dir: Path | None = None,
) -> None:
    """
    Processes one zip at a time to minimise peak disk usage.

    For each archive:
      1. Extract into extract_dir.
      2. Run the pipeline and gender check, save an interim CSV.
      3. Delete the extracted folder contents.

    After all archives are processed, merges the interim CSVs into a single
    final summary, computes the true missing cases across the full dataset,
    and writes accumulated gender data back to classes.csv.
    """
    out_dir = log_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = out_dir / "interim"
    interim_dir.mkdir(exist_ok=True)

    logger = setup_logger(str(out_dir / "inventory.log"))
    logger.info("=== Batch Extraction & Inventory Scan (partitioned mode) ===")

    if not zip_dir.exists():
        raise RuntimeError(f"Zip source directory not found: '{zip_dir}'")

    zip_files = _find_zips(zip_dir, logger)
    if not zip_files:
        return

    mixed_csv_paths: list[Path] = []
    all_rows: list[dict] = []

    for i, zip_path in enumerate(zip_files, start=1):
        logger.info(f"\n[{i}/{len(zip_files)}] Processing: {zip_path.name}")

        # Clean extract_dir before each zip to reclaim space.
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        if not _extract_zip(zip_path, extract_dir, logger):
            logger.warning(f"  Skipping pipeline for {zip_path.name} (extraction failed).")
            continue

        mixed_csv = interim_dir / f"mixed_{zip_path.stem}.csv"
        mixed_csv_paths.append(mixed_csv)

        batch_rows = _run_pipeline_steps(extract_dir, mixed_csv, logger)
        all_rows.extend(batch_rows)

        # Save interim CSV so progress survives an interruption.
        interim_csv = interim_dir / f"{zip_path.stem}.csv"
        _write_summary_csv(batch_rows, interim_csv)
        logger.info(f"  Interim CSV: '{interim_csv}' ({len(batch_rows)} rows)")

        # Wipe extracted data — no longer needed.
        shutil.rmtree(extract_dir)
        logger.info(f"  Cleaned up '{extract_dir}'.")

    # Compute true missing cases now that we have seen all batches and join
    # class data so their class/location/patient_age fields are populated.
    missing_rows = find_missing_cases(all_rows)
    for row in missing_rows:
        row["validation_status"] = "MISSING"
    missing_rows = join_class_data(missing_rows, str(CLASSES_CSV_PATH))
    all_rows.extend(missing_rows)

    # Class and gender checks run once over the full merged dataset so that
    # filesystem-MISSING cases are also evaluated against classes.csv.
    add_class_status(all_rows, CLASSES_CSV_PATH, logger)
    check_missing_gender(all_rows, logger)
    write_gender_to_classes_csv(all_rows, CLASSES_CSV_PATH, logger)

    # Merge mixed-series CSVs from all batches.
    _merge_mixed_csvs(mixed_csv_paths, out_dir / "mixed_series_analysis.csv")

    final_csv = out_dir / "inventory_summary.csv"
    _write_summary_csv(all_rows, final_csv)
    logger.info(f"\nFinal summary CSV written to: '{final_csv}' ({len(all_rows)} rows)")
    _log_status_breakdown(all_rows, logger)
    logger.info("\nInventory scan complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract zip archives and run an inventory scan without NIfTI conversion."
    )
    parser.add_argument(
        "--zip-dir",
        default=str(DEFAULT_ZIP_DIR),
        help=f"Directory containing .zip archives (default: {DEFAULT_ZIP_DIR})",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(DEFAULT_EXTRACT_DIR),
        help=f"Directory to extract archives into (default: {DEFAULT_EXTRACT_DIR})",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help=(
            "Process one zip at a time, deleting extracted files after each scan. "
            "Use when local disk space is limited."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="(Full mode only) Skip extraction and scan the extract directory as-is.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            f"Directory for output files (default: data_engine/output/inventory/). "
            "Override to keep separate runs from overwriting each other."
        ),
    )
    args = parser.parse_args()

    if args.partitioned:
        if args.skip_extract:
            parser.error("--skip-extract is not compatible with --partitioned.")
        run_partitioned(
            zip_dir=Path(args.zip_dir),
            extract_dir=Path(args.extract_dir),
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )
    else:
        run_inventory(
            zip_dir=Path(args.zip_dir),
            extract_dir=Path(args.extract_dir),
            skip_extract=args.skip_extract,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )
