"""Find duplicate exams by hashing image data.

Each exam is read once and its voxel/pixel data reduced to an MD5 fingerprint;
exams with the same fingerprint are byte-identical images. No pairwise loop.

  nifti  hash every converted NIfTI (/mnt/data/nifti), print duplicate groups
         and whether a member is in the test set
  dicom  for the given exams, hash the raw DICOM pixel data of the series step 1
         converted (dcm_path + series_number in reports/step1_convert.csv), and
         print the DICOM metadata next to it, to tell "same pixels" from "same
         metadata"

Usage:
    python scripts/find_duplicates.py nifti
    python scripts/find_duplicates.py dicom BP0074 BP0464 BP0112 --random 1
"""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "step1_convert.csv"
TEST_IDS = ROOT.parent / "dataset" / "test_ids.json"
META = ["PatientID", "PatientAge", "PatientSex", "StudyDate", "AcquisitionTime",
        "ManufacturerModelName", "StudyInstanceUID", "SeriesInstanceUID"]


def nifti_hash(path: str) -> tuple[str, str]:
    return Path(path).name[:-7], hashlib.md5(np.asarray(nib.load(path).dataobj).tobytes()).hexdigest()


def dicom_series(dcm_dir: Path, series_number: int) -> list[pydicom.Dataset]:
    """The converted series' slices, sorted along the slice normal."""
    slices = []
    for f in sorted(p for p in dcm_dir.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(f)
        except Exception:
            continue  # non-DICOM files (DICOMDIR, thumbnails)
        if int(getattr(ds, "SeriesNumber", -1)) == series_number and "PixelData" in ds:
            slices.append(ds)
    if not slices:
        raise ValueError(f"no slices of series {series_number} in {dcm_dir}")
    normal = np.cross(*np.reshape(np.array(slices[0].ImageOrientationPatient, float), (2, 3)))
    return sorted(slices, key=lambda s: float(np.dot(normal, np.array(s.ImagePositionPatient, float))))


def dicom_row(exam: str, row: pd.Series) -> dict:
    slices = dicom_series(Path(row.dcm_path), int(row.series_number))
    pixels = hashlib.md5()
    for s in slices:
        pixels.update(s.pixel_array.tobytes())  # decoded pixels: same image under any transfer syntax
    return {"exam": exam, "class": int(row["class"]), "slices": len(slices),
            "pixel_md5": pixels.hexdigest()[:12], "folder": row.original_name,
            **{k: str(getattr(slices[0], k, "")) for k in META}}


def cmd_nifti(args) -> None:
    files = sorted(str(p) for p in Path(args.nifti).glob("*/*.nii.gz"))
    groups = defaultdict(list)
    with ProcessPoolExecutor(args.workers) as pool:
        for exam, h in pool.map(nifti_hash, files, chunksize=8):
            groups[h].append(exam)
    test = set(json.loads(TEST_IDS.read_text()))
    dups = [g for g in groups.values() if len(g) > 1]
    print(f"{len(files)} NIfTIs, {len(dups)} duplicate groups")
    for g in dups:
        print("  " + "  ".join(f"{e}{' (TEST)' if e in test else ''}" for e in g))


def cmd_dicom(args) -> None:
    report = pd.read_csv(REPORT)
    report = report[report.conversion_status == "converted"].set_index("exam_id")
    exams = list(args.exams)
    if args.random:
        rest = sorted(set(report.index) - set(exams))
        exams += random.Random(args.seed).sample(rest, args.random)
    missing = [e for e in exams if e not in report.index]
    if missing:
        sys.exit(f"ERROR: not converted / unknown: {missing}")
    rows = [dicom_row(e, report.loc[e]) for e in exams]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 24)
    print(df.drop(columns=["StudyInstanceUID", "SeriesInstanceUID"]).to_string(index=False))
    print("\nidentical pixel data:")
    for h, g in df.groupby("pixel_md5").exam:
        if len(g) > 1:
            same = [k for k in META if df.loc[g.index, k].nunique() == 1]
            print(f"  {' = '.join(g)}  (same metadata: {', '.join(same) or 'none'})")
    print("unique:", ", ".join(df.exam[~df.pixel_md5.duplicated(keep=False)]) or "none")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nifti")
    n.add_argument("--nifti", default="/mnt/data/nifti")
    n.add_argument("--workers", type=int, default=4)
    d = sub.add_parser("dicom")
    d.add_argument("exams", nargs="*")
    d.add_argument("--random", type=int, default=0, help="add N random other exams as controls")
    d.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"nifti": cmd_nifti, "dicom": cmd_dicom}[args.cmd](args)


if __name__ == "__main__":
    main()
