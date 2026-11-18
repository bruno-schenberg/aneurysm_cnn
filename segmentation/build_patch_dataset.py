"""Build a 3D patch dataset from extracted candidates (Phase 7A, stage 2 input).

Crops a fixed-size cube around each candidate centroid from its source
volume (the same NIfTI nnU-Net ran on), and writes it out in the same
class-directory convention every other training_engine dataset uses
(`<output_dir>/0/*.nii.gz`, `<output_dir>/1/*.nii.gz`), so it plugs straight
into the existing `get_data_list`/`build_dataloaders` pipeline unchanged.

Usage:
    python build_patch_dataset.py \\
        --candidates-csv segmentation/candidates.csv \\
        --output-dir ~/data/datasets/patch_dataset_64 \\
        --patch-size 64
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_candidates(candidates_csv: str) -> list[dict]:
    with open(candidates_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    # Only rows with a definite patch_label (excludes ambiguous multi-candidate positives)
    return [r for r in rows if r["patch_label"] in ("0", "1")]


def _crop_patch(volume: np.ndarray, center: tuple[float, float, float], size: int) -> np.ndarray:
    """Crop a size^3 cube centered at `center` (voxel coords), zero-padding at volume edges."""
    patch = np.zeros((size, size, size), dtype=volume.dtype)
    half = size // 2

    starts = [int(round(center[dim])) - half for dim in range(3)]
    ends = [start + size for start in starts]

    vol_slices, patch_slices = [], []
    for dim in range(3):
        vol_start = max(starts[dim], 0)
        vol_end = min(ends[dim], volume.shape[dim])
        patch_start = vol_start - starts[dim]
        patch_end = patch_start + max(vol_end - vol_start, 0)
        vol_slices.append(slice(vol_start, vol_end))
        patch_slices.append(slice(patch_start, patch_end))

    if all(s.stop > s.start for s in vol_slices):
        patch[tuple(patch_slices)] = volume[tuple(vol_slices)]
    return patch


def build_patch_dataset(candidates_csv: str, output_dir: str, patch_size: int) -> None:
    candidates = _load_candidates(candidates_csv)
    out_root = Path(output_dir)
    (out_root / "0").mkdir(parents=True, exist_ok=True)
    (out_root / "1").mkdir(parents=True, exist_ok=True)

    by_exam: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_exam[row["exam_id"]].append(row)

    n_written = 0
    n_missing_source = 0
    for exam_id, rows in sorted(by_exam.items()):
        source_path = rows[0]["source"]
        if not Path(source_path).exists():
            n_missing_source += 1
            logger.warning("Source volume not found for %s: %s", exam_id, source_path)
            continue

        img = nib.load(source_path)
        volume = img.get_fdata().astype(np.float32)
        affine = img.affine

        for row in rows:
            center = (float(row["cz"]), float(row["cy"]), float(row["cx"]))
            patch = _crop_patch(volume, center, patch_size)

            label = row["patch_label"]
            out_path = out_root / label / f"{exam_id}_{row['component_id']}.nii.gz"
            nib.save(nib.Nifti1Image(patch, affine), str(out_path))
            n_written += 1

    n_pos = len(list((out_root / "1").glob("*.nii.gz")))
    n_neg = len(list((out_root / "0").glob("*.nii.gz")))
    logger.info("Wrote %d patches (%d positive, %d negative) to %s", n_written, n_pos, n_neg, out_root)
    if n_missing_source:
        logger.warning("%d exams skipped — source volume missing", n_missing_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", default="segmentation/candidates.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=64)
    args = parser.parse_args()
    build_patch_dataset(args.candidates_csv, args.output_dir, args.patch_size)


if __name__ == "__main__":
    main()
