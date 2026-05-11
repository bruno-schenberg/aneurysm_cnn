"""
Temporary script: extract PatientSex from DICOM headers in zip files at
/mnt/data/raw and write the results as a 'gender' column in
data_engine/dataset/classes.csv.

One zip is unzipped at a time; the extracted folder is deleted before the
next zip is opened, so the drive never holds more than one batch at once.

Usage (from repo root or data_engine/):
    python data_engine/extract_gender.py

Gender encoding: M -> 0, F -> 1
Rows with no matching case in classes.csv or unreadable sex tags are skipped
and reported at the end. Re-running is safe: existing gender values are not
overwritten unless a fresh read succeeds.
"""

import csv
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError


# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR    = Path("/mnt/data/raw")
CLASSES_CSV = Path(__file__).resolve().parent / "dataset" / "classes.csv"

SEX_TO_GENDER = {"M": 0, "F": 1}


# ── Helpers ────────────────────────────────────────────────────────────────────

_BP_PATTERN = re.compile(r"^bp_?(\d+)", re.IGNORECASE)


def normalise_exam_name(name: str) -> str | None:
    """Map a raw folder name (e.g. 'bp001', 'BP_12') to 'BP{NNN}', or None."""
    m = _BP_PATTERN.match(name)
    return f"BP{int(m.group(1)):03d}" if m else None


def find_exam_folders(root: Path) -> list[Path]:
    """
    Return all directories under *root* whose name matches the BP{NNN} pattern.

    Searches up to two levels deep so the script handles both:
      root/bp001/          (zip extracts exam folders at top level)
      root/batch/bp001/    (zip has a single top-level batch folder)
    """
    candidates: list[Path] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if normalise_exam_name(entry.name):
            candidates.append(entry)
        else:
            # One level deeper — handles a batch wrapper folder
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and normalise_exam_name(sub.name):
                    candidates.append(sub)

    return candidates


def find_first_dcm(folder: Path) -> Path | None:
    """
    Recursively find the first .dcm file inside *folder*.

    Using os.walk instead of rglob so we can break early on the first hit
    without scanning the entire tree.
    """
    for dirpath, _dirs, filenames in os.walk(folder):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                return Path(dirpath) / fname
    return None


def read_patient_sex(dcm_path: Path) -> str | None:
    """
    Return the PatientSex string from a DICOM header, or None on failure.

    stop_before_pixels=True skips the (potentially large) pixel data so only
    the header tags are loaded — much faster for a metadata-only scan.
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        val = getattr(ds, "PatientSex", None)
        if val is not None:
            return str(val).strip().upper()
    except (InvalidDicomError, Exception):
        pass
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CLASSES_CSV.exists():
        print(f"ERROR: {CLASSES_CSV} not found.")
        sys.exit(1)

    # Read existing classes.csv preserving all rows and column order
    with open(CLASSES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames: list[str] = list(reader.fieldnames or [])
        rows = list(reader)

    row_by_exam: dict[str, int] = {row["exam"]: i for i, row in enumerate(rows)}

    # gender_results accumulates across all zips before we touch the CSV
    gender_results: dict[str, int] = {}
    skipped: list[str] = []         # exams where sex tag was absent / unreadable
    not_in_csv: list[str] = []      # exam folders with no matching row in CSV

    zip_files = sorted(RAW_DIR.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in {RAW_DIR}")
        sys.exit(0)

    print(f"Found {len(zip_files)} zip file(s) in {RAW_DIR}\n")

    for zip_path in zip_files:
        print(f"→ {zip_path.name}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="gender_ext_", dir=RAW_DIR.parent))

        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)

            exam_folders = find_exam_folders(tmp_dir)
            if not exam_folders:
                print(f"  No BP{{NNN}} exam folders found — skipping")
                continue

            found_count = 0
            for exam_folder in exam_folders:
                canonical = normalise_exam_name(exam_folder.name)
                if canonical is None:
                    continue

                if canonical not in row_by_exam:
                    not_in_csv.append(canonical)
                    continue

                # Don't re-read if we already have a value from a previous zip
                if canonical in gender_results:
                    continue

                dcm_file = find_first_dcm(exam_folder)
                if dcm_file is None:
                    skipped.append(f"{canonical} (no .dcm found)")
                    continue

                sex = read_patient_sex(dcm_file)
                if sex not in SEX_TO_GENDER:
                    skipped.append(f"{canonical} (PatientSex='{sex}')")
                    continue

                gender_results[canonical] = SEX_TO_GENDER[sex]
                found_count += 1

            print(f"  {found_count}/{len(exam_folders)} exams with gender extracted")

        except zipfile.BadZipFile as e:
            print(f"  ERROR reading zip: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Write back to classes.csv ──────────────────────────────────────────────

    if not gender_results:
        print("\nNo gender data extracted — classes.csv unchanged.")
        return

    if "gender" not in fieldnames:
        fieldnames.append("gender")

    # Merge: only write new values; don't overwrite a value already in the CSV
    for canonical, gender_val in gender_results.items():
        i = row_by_exam[canonical]
        existing = str(rows[i].get("gender", "")).strip()
        if not existing:
            rows[i]["gender"] = gender_val

    with open(CLASSES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────

    filled = sum(1 for r in rows if str(r.get("gender", "")).strip() != "")
    total  = len(rows)
    print(f"\nDone — {len(gender_results)} new gender value(s) written to {CLASSES_CSV}")
    print(f"CSV coverage: {filled}/{total} rows have gender")

    if skipped:
        print(f"\nSkipped ({len(skipped)} exams — sex tag absent or not M/F):")
        for s in skipped:
            print(f"  {s}")

    if not_in_csv:
        unique = sorted(set(not_in_csv))
        print(f"\nNot in classes.csv ({len(unique)} exam(s)):")
        print("  " + ", ".join(unique[:20]) + (" ..." if len(unique) > 20 else ""))


if __name__ == "__main__":
    main()
