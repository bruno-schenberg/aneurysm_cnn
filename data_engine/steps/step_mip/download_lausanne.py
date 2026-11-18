"""Download and organize the Lausanne TOF-MRA Aneurysm Cohort (OpenNeuro ds003949).

Downloads TOF-MRA volumes via AWS S3 (no auth required), reads participants.tsv
for labels, and organizes into our standard 0/ 1/ class structure.

Output:
    <output_root>/lausanne_raw/          — raw BIDS download
    <output_root>/lausanne_nifti/0/      — control (no aneurysm)
    <output_root>/lausanne_nifti/1/      — patient (has aneurysm)

Usage:
    python -m steps.step_mip.download_lausanne --output-root /path/to/data
    python -m steps.step_mip.download_lausanne --output-root /path/to/data --auto-install
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_ID = "ds003949"
S3_BUCKET = f"s3://openneuro.org/{DATASET_ID}"


def _ensure_awscli(auto_install: bool) -> str:
    """Ensure aws CLI is available. Returns the command name.

    Raises RuntimeError (rather than silently installing) unless
    auto_install is True — a pip install as a side effect of running this
    script is surprising on a shared/locked-down cluster environment.
    """
    if shutil.which("aws"):
        return "aws"

    if not auto_install:
        raise RuntimeError(
            "awscli not found on PATH. Install manually (pip install awscli) "
            "or re-run with --auto-install to have this script install it."
        )

    print("awscli not found on PATH, installing via pip (--auto-install)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "awscli"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install awscli. Install manually: pip install awscli"
        )

    # After pip install, the `aws` binary lands in the same bin/ as python
    aws_path = Path(sys.executable).parent / "aws"
    if aws_path.exists():
        return str(aws_path)
    if shutil.which("aws"):
        return "aws"
    raise RuntimeError(
        "awscli installed but 'aws' not found on PATH. "
        "Try: pip install awscli && hash -r"
    )


def download_from_s3(raw_dir: Path, auto_install: bool = False) -> None:
    """Download TOF-MRA volumes + participants.tsv from OpenNeuro S3."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    aws_cmd = _ensure_awscli(auto_install)

    cmd = [
        aws_cmd, "s3", "sync", "--no-sign-request",
        S3_BUCKET, str(raw_dir),
        "--exclude", "*",
        "--include", "participants.tsv",
        "--include", "participants.json",
        "--include", "*/anat/*_angio.nii.gz",
    ]
    print(f"Downloading from {S3_BUCKET}...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed (exit code {result.returncode})")


def read_labels(raw_dir: Path) -> dict[str, str]:
    """Read participants.tsv → {participant_id: 'patient'|'control'}."""
    tsv_path = raw_dir / "participants.tsv"
    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found", file=sys.stderr)
        sys.exit(1)

    labels = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["participant_id"].strip()
            group = row["group"].strip().lower()
            labels[pid] = group
    return labels


def organize(raw_dir: Path, nifti_dir: Path) -> None:
    """Copy TOF-MRA volumes into 0/ (control) and 1/ (patient) class dirs."""
    labels = read_labels(raw_dir)
    group_to_class = {"control": "0", "patient": "1"}

    for cls in ("0", "1"):
        (nifti_dir / cls).mkdir(parents=True, exist_ok=True)

    # Find all angio NIfTIs
    angio_files = sorted(raw_dir.rglob("*_angio.nii.gz"))
    print(f"Found {len(angio_files)} TOF-MRA volumes")

    copied = {"0": 0, "1": 0}
    skipped = 0

    for fpath in angio_files:
        # Extract participant ID from path (sub-XXX)
        parts = fpath.parts
        sub_id = None
        for p in parts:
            if p.startswith("sub-"):
                sub_id = p
                break

        if sub_id is None or sub_id not in labels:
            print(f"  WARNING: Could not find label for {fpath.name}, skipping")
            skipped += 1
            continue

        group = labels[sub_id]
        cls = group_to_class.get(group)
        if cls is None:
            print(f"  WARNING: Unknown group '{group}' for {sub_id}, skipping")
            skipped += 1
            continue

        # Use participant ID as filename (simpler than BIDS path)
        # Extract session from path for uniqueness
        ses_id = None
        for p in parts:
            if p.startswith("ses-"):
                ses_id = p
                break

        out_name = f"{sub_id}"
        if ses_id:
            out_name += f"_{ses_id}"
        out_name += ".nii.gz"

        dst = nifti_dir / cls / out_name
        if not dst.exists():
            shutil.copy2(fpath, dst)
        copied[cls] += 1

    print(f"\nOrganized: {copied['0']} control (class 0), {copied['1']} patient (class 1)")
    if skipped:
        print(f"Skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Lausanne TOF-MRA dataset")
    parser.add_argument(
        "--output-root", required=True,
        help="Root dir (will create lausanne_raw/ and lausanne_nifti/ subdirs)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip S3 download, only re-organize existing files",
    )
    parser.add_argument(
        "--auto-install", action="store_true",
        help="Auto-install awscli via pip if not found on PATH (default: error and tell you to install it)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    raw_dir = output_root / "lausanne_raw"
    nifti_dir = output_root / "lausanne_nifti"

    if not args.skip_download:
        download_from_s3(raw_dir, auto_install=args.auto_install)
    else:
        print("Skipping download (--skip-download)")

    print("\nOrganizing into class directories...")
    organize(raw_dir, nifti_dir)

    print("\nDone.")
    for cls, label in [("0", "control"), ("1", "patient")]:
        count = len(list((nifti_dir / cls).glob("*.nii.gz")))
        print(f"  {nifti_dir / cls}: {count} files ({label})")


if __name__ == "__main__":
    main()
