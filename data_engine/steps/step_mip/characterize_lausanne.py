"""Characterize the Lausanne dataset and compare with our data.

Queries NIfTI headers and BIDS sidecar JSONs on the cluster.
Also downloads missing sidecar JSONs if --download-jsons is passed.

Usage:
    # Just characterize (assumes data already on cluster):
    python -m steps.step_mip.characterize_lausanne --lausanne-raw ~/data/lausanne_raw \
        --lausanne-nifti ~/data/lausanne_nifti --ours ~/data/datasets/c90z_384

    # Also download missing BIDS sidecar JSONs first:
    python -m steps.step_mip.characterize_lausanne --download-jsons --lausanne-raw ~/data/lausanne_raw \
        --lausanne-nifti ~/data/lausanne_nifti --ours ~/data/datasets/c90z_384
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def download_sidecar_jsons(raw_dir: Path) -> None:
    """Download BIDS sidecar JSONs from OpenNeuro S3."""
    import shutil

    aws_cmd = shutil.which("aws") or "aws"
    cmd = [
        aws_cmd, "s3", "sync", "--no-sign-request",
        "s3://openneuro.org/ds003949", str(raw_dir),
        "--exclude", "*",
        "--include", "*/anat/*_angio.json",
    ]
    print(f"Downloading sidecar JSONs...")
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def scan_nifti_headers(data_dir: Path, label: str) -> list[dict]:
    """Read NIfTI headers without loading voxel data."""
    records = []
    for cls in ("0", "1"):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for fpath in sorted(cls_dir.glob("*.nii.gz")):
            img = nib.load(str(fpath))
            zooms = img.header.get_zooms()[:3]
            records.append({
                "dataset": label,
                "file": fpath.name,
                "class": cls,
                "shape_x": img.shape[0],
                "shape_y": img.shape[1],
                "shape_z": img.shape[2],
                "spacing_x": round(float(zooms[0]), 4),
                "spacing_y": round(float(zooms[1]), 4),
                "spacing_z": round(float(zooms[2]), 4),
                "dtype": str(img.header.get_data_dtype()),
            })
    return records


def scan_sidecar_jsons(raw_dir: Path) -> list[dict]:
    """Read BIDS sidecar JSONs for acquisition parameters."""
    records = []
    for jpath in sorted(raw_dir.rglob("*_angio.json")):
        with open(jpath) as f:
            data = json.load(f)
        # Extract subject ID
        sub_id = None
        for part in jpath.parts:
            if part.startswith("sub-"):
                sub_id = part
                break
        records.append({
            "subject": sub_id,
            "MagneticFieldStrength": data.get("MagneticFieldStrength"),
            "Manufacturer": data.get("Manufacturer"),
            "ManufacturersModelName": data.get("ManufacturersModelName"),
            "RepetitionTime": data.get("RepetitionTime"),
            "EchoTime": data.get("EchoTime"),
            "FlipAngle": data.get("FlipAngle"),
            "PixelBandwidth": data.get("PixelBandwidth"),
            "MRAcquisitionType": data.get("MRAcquisitionType"),
        })
    return records


def print_nifti_summary(records: list[dict], label: str) -> None:
    """Print summary statistics for a set of NIfTI headers."""
    if not records:
        print(f"\n{label}: no files found")
        return

    n = len(records)
    n_pos = sum(1 for r in records if r["class"] == "1")
    n_neg = sum(1 for r in records if r["class"] == "0")

    shapes = np.array([[r["shape_x"], r["shape_y"], r["shape_z"]] for r in records])
    spacings = np.array([[r["spacing_x"], r["spacing_y"], r["spacing_z"]] for r in records])

    print(f"\n{'='*60}")
    print(f"{label}: {n} volumes ({n_pos} positive, {n_neg} negative)")
    print(f"{'='*60}")
    print(f"  Class balance: {n_pos/n*100:.1f}% positive")
    print(f"  Shape X:   min={shapes[:,0].min():.0f}  max={shapes[:,0].max():.0f}  "
          f"median={np.median(shapes[:,0]):.0f}  std={shapes[:,0].std():.1f}")
    print(f"  Shape Y:   min={shapes[:,1].min():.0f}  max={shapes[:,1].max():.0f}  "
          f"median={np.median(shapes[:,1]):.0f}  std={shapes[:,1].std():.1f}")
    print(f"  Shape Z:   min={shapes[:,2].min():.0f}  max={shapes[:,2].max():.0f}  "
          f"median={np.median(shapes[:,2]):.0f}  std={shapes[:,2].std():.1f}")
    print(f"  Spacing X: min={spacings[:,0].min():.4f}  max={spacings[:,0].max():.4f}  "
          f"median={np.median(spacings[:,0]):.4f}")
    print(f"  Spacing Y: min={spacings[:,1].min():.4f}  max={spacings[:,1].max():.4f}  "
          f"median={np.median(spacings[:,1]):.4f}")
    print(f"  Spacing Z: min={spacings[:,2].min():.4f}  max={spacings[:,2].max():.4f}  "
          f"median={np.median(spacings[:,2]):.4f}")

    # Unique spacings
    unique_spacings = np.unique(spacings, axis=0)
    if len(unique_spacings) <= 5:
        print(f"  Unique spacings ({len(unique_spacings)}):")
        for s in unique_spacings:
            count = np.sum(np.all(spacings == s, axis=1))
            print(f"    ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}) — {count} volumes")
    else:
        print(f"  {len(unique_spacings)} unique spacings (variable resolution)")

    # Unique shapes
    unique_shapes = np.unique(shapes, axis=0)
    if len(unique_shapes) <= 5:
        print(f"  Unique shapes ({len(unique_shapes)}):")
        for s in unique_shapes:
            count = np.sum(np.all(shapes == s, axis=1))
            print(f"    ({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) — {count} volumes")
    else:
        print(f"  {len(unique_shapes)} unique shapes (variable matrix size)")

    dtypes = set(r["dtype"] for r in records)
    print(f"  Data types: {dtypes}")


def print_sidecar_summary(records: list[dict]) -> None:
    """Print summary of BIDS sidecar JSON acquisition parameters."""
    if not records:
        print("\nNo BIDS sidecar JSONs found.")
        print("Run with --download-jsons to fetch them from OpenNeuro.")
        return

    print(f"\n{'='*60}")
    print(f"BIDS acquisition parameters ({len(records)} sidecar JSONs)")
    print(f"{'='*60}")

    for key in ["MagneticFieldStrength", "Manufacturer", "ManufacturersModelName",
                 "RepetitionTime", "EchoTime", "FlipAngle", "MRAcquisitionType"]:
        values = [r[key] for r in records if r[key] is not None]
        if not values:
            print(f"  {key}: not available")
            continue

        if isinstance(values[0], (int, float)):
            arr = np.array(values, dtype=float)
            unique = np.unique(arr)
            if len(unique) <= 5:
                print(f"  {key}: {', '.join(str(v) for v in unique)} "
                      f"({len(values)}/{len(records)} files)")
            else:
                print(f"  {key}: min={arr.min():.4f}  max={arr.max():.4f}  "
                      f"median={np.median(arr):.4f}")
        else:
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            for v, c in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {key}: {v} ({c}/{len(records)})")


def print_participants_tsv(raw_dir: Path) -> None:
    """Print participants.tsv summary."""
    tsv = raw_dir / "participants.tsv"
    if not tsv.exists():
        print("\nparticipants.tsv: not found")
        return

    with open(tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    print(f"\n{'='*60}")
    print(f"participants.tsv ({len(rows)} subjects)")
    print(f"{'='*60}")
    print(f"  Columns: {', '.join(rows[0].keys())}")

    groups = {}
    for r in rows:
        g = r.get("group", "unknown")
        groups[g] = groups.get(g, 0) + 1
    print(f"  Groups: {groups}")

    # Age/sex if available
    if "age" in rows[0]:
        ages = [float(r["age"]) for r in rows if r.get("age")]
        print(f"  Age: min={min(ages):.0f}  max={max(ages):.0f}  "
              f"mean={np.mean(ages):.1f}±{np.std(ages):.1f}")
        for g in sorted(groups):
            g_ages = [float(r["age"]) for r in rows if r.get("group") == g and r.get("age")]
            if g_ages:
                print(f"    {g}: {np.mean(g_ages):.1f}±{np.std(g_ages):.1f} (n={len(g_ages)})")

    if "sex" in rows[0]:
        sexes = {}
        for r in rows:
            s = r.get("sex", "unknown")
            sexes[s] = sexes.get(s, 0) + 1
        print(f"  Sex: {sexes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize Lausanne vs our dataset")
    parser.add_argument("--lausanne-raw", required=True, help="Path to lausanne_raw/ (BIDS)")
    parser.add_argument("--lausanne-nifti", required=True, help="Path to lausanne_nifti/ (0/1)")
    parser.add_argument("--ours", required=True, help="Path to our dataset (e.g. c90z_384)")
    parser.add_argument("--download-jsons", action="store_true",
                        help="Download missing BIDS sidecar JSONs from OpenNeuro S3")
    args = parser.parse_args()

    lausanne_raw = Path(args.lausanne_raw)
    lausanne_nifti = Path(args.lausanne_nifti)
    ours = Path(args.ours)

    if args.download_jsons:
        download_sidecar_jsons(lausanne_raw)

    # Participants metadata
    print_participants_tsv(lausanne_raw)

    # BIDS sidecar JSONs
    sidecars = scan_sidecar_jsons(lausanne_raw)
    print_sidecar_summary(sidecars)

    # NIfTI headers
    print("\nScanning Lausanne NIfTI headers...")
    laus_records = scan_nifti_headers(lausanne_nifti, "Lausanne")
    print_nifti_summary(laus_records, "Lausanne")

    print("\nScanning our NIfTI headers...")
    ours_records = scan_nifti_headers(ours, "Our data (c90z_384)")
    print_nifti_summary(ours_records, "Our data (c90z_384)")


if __name__ == "__main__":
    main()
