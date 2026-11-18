"""Prepare NIfTI files for nnU-Net inference.

Creates symlinks in nnU-Net format (<case_id>_0000.nii.gz) from either:
  - Original NIfTIs: /path/to/nifti/{0,1}/<exam_id>.nii.gz
  - Resampled dataset: /path/to/datasets/<variant>/{0,1}/<exam_id>.nii.gz

Saves a mapping CSV for evaluation.

Usage:
    python prepare_input.py /path/to/data /path/to/output [--dataset-csv dataset.csv]
"""

import argparse
import csv
from pathlib import Path


def prepare(source_dir: str, output_dir: str, dataset_csv: str | None = None):
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(src.rglob("*.nii.gz"))
    if not nifti_files:
        print(f"ERROR: No .nii.gz files found in {src}")
        return

    # Load labels from dataset.csv if provided
    labels = {}
    if dataset_csv:
        with open(dataset_csv) as f:
            for row in csv.DictReader(f):
                if row["class"].strip():
                    labels[row["exam"]] = int(float(row["class"]))

    # Detect class-organized layout (has 0/ or 1/ subdirs)
    has_class_dirs = (src / "0").is_dir() or (src / "1").is_dir()

    mapping = []
    for nii_path in nifti_files:
        exam_id = nii_path.name.replace(".nii.gz", "")

        if has_class_dirs:
            label = int(nii_path.parent.name)
        elif exam_id in labels:
            label = labels[exam_id]
        else:
            label = -1

        link_name = out / f"{exam_id}_0000.nii.gz"
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        link_name.symlink_to(nii_path.resolve())

        mapping.append({"exam_id": exam_id, "label": label, "source": str(nii_path)})

    # Save mapping alongside input dir
    mapping_csv = out.parent / "mapping.csv"
    with open(mapping_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exam_id", "label", "source"])
        writer.writeheader()
        writer.writerows(mapping)

    pos = sum(1 for m in mapping if m["label"] == 1)
    neg = sum(1 for m in mapping if m["label"] == 0)
    unk = len(mapping) - pos - neg
    print(f"Prepared {len(mapping)} files in {out}")
    print(f"  Class 0: {neg}, Class 1: {pos}, Unknown: {unk}")
    print(f"Mapping saved to {mapping_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NIfTI files for nnU-Net inference")
    parser.add_argument("source_dir", help="Directory with NIfTI files (class-organized or flat)")
    parser.add_argument("output_dir", help="Output directory for nnU-Net input symlinks")
    parser.add_argument("--dataset-csv", help="Path to dataset.csv for labels (if source is flat)")
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.dataset_csv)
