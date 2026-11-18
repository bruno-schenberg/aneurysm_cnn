"""Extract candidate lesion centroids from nnU-Net segmentation masks (Phase 7A).

Stage 1 of the two-stage cascade: nnU-Net's predicted masks contain one or
more connected components per exam. This script reduces each mask to a list
of (centroid, voxel_count) candidates and assigns each a patch-level label
for stage 2 (the patch classifier):

  - Exam is volume-level NEGATIVE -> every candidate is a false positive
    (label 0). This is exact: a negative exam has no aneurysm anywhere, so
    every ROI nnU-Net found is by definition wrong.
  - Exam is volume-level POSITIVE with exactly ONE candidate -> that
    candidate is the aneurysm (label 1). Also exact: nnU-Net found exactly
    one thing in a volume known to contain exactly the kind of lesion it's
    looking for.
  - Exam is volume-level POSITIVE with MULTIPLE candidates -> WHICH one is
    the real aneurysm is unknown without per-lesion ground truth (we only
    have volume-level labels). These are excluded from the patch dataset rather than
    guessed at (labeling the wrong one as positive would inject silent,
    confident label noise). Logged separately so you can see how much data
    this costs.
  - Exam is volume-level POSITIVE with ZERO candidates -> nnU-Net missed it
    (false negative at stage 1). Nothing to extract; this is an inherent
    ceiling of the cascade approach (stage 2 can't recover a miss stage 1
    never proposed).

Usage:
    python extract_candidates.py \\
        --masks-dir segmentation/nnunet_output \\
        --mapping-csv segmentation/mapping.csv \\
        --output-csv segmentation/candidates.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Full 26-connectivity: diagonally-touching voxels merge into one component,
# so a single irregularly-shaped lesion isn't fragmented into several.
_STRUCTURE = ndimage.generate_binary_structure(3, 3)


def _load_mapping(mapping_csv: str) -> list[dict]:
    with open(mapping_csv, newline="") as f:
        return list(csv.DictReader(f))


def extract_candidates(masks_dir: str, mapping_csv: str) -> list[dict]:
    masks_path = Path(masks_dir)
    mapping = _load_mapping(mapping_csv)

    rows: list[dict] = []
    n_missing_mask = 0
    n_unknown_label = 0
    n_excluded_ambiguous = 0
    n_fn = 0

    for entry in mapping:
        exam_id = entry["exam_id"]
        label = int(entry["label"])
        source = entry["source"]

        if label == -1:
            n_unknown_label += 1
            continue

        mask_file = masks_path / f"{exam_id}.nii.gz"
        if not mask_file.exists():
            n_missing_mask += 1
            continue

        mask = nib.load(str(mask_file)).get_fdata()
        binary = mask > 0

        labeled, n_components = ndimage.label(binary, structure=_STRUCTURE)
        if n_components == 0:
            if label == 1:
                n_fn += 1
            continue

        centroids = ndimage.center_of_mass(binary, labeled, range(1, n_components + 1))
        voxel_counts = ndimage.sum(binary, labeled, range(1, n_components + 1))

        ambiguous = label == 1 and n_components > 1
        if ambiguous:
            n_excluded_ambiguous += n_components

        for component_id, (centroid, voxel_count) in enumerate(zip(centroids, voxel_counts), start=1):
            if label == 0:
                patch_label = 0
            elif ambiguous:
                patch_label = ""
            else:
                patch_label = 1

            rows.append({
                "exam_id": exam_id,
                "component_id": component_id,
                "cz": round(centroid[0], 1),
                "cy": round(centroid[1], 1),
                "cx": round(centroid[2], 1),
                "voxel_count": int(voxel_count),
                "true_volume_label": label,
                "n_components_in_exam": n_components,
                "patch_label": patch_label,
                "source": source,
            })

    logger.info("Processed %d exams from %s", len(mapping), mapping_csv)
    logger.info("  missing mask file: %d", n_missing_mask)
    logger.info("  unknown volume label (skipped): %d", n_unknown_label)
    logger.info("  positive exam, zero candidates (stage-1 false negative): %d", n_fn)
    n_pos = sum(1 for r in rows if r["patch_label"] == 1)
    n_neg = sum(1 for r in rows if r["patch_label"] == 0)
    logger.info("  candidates -> patch_label=1 (unambiguous TP): %d", n_pos)
    logger.info("  candidates -> patch_label=0 (FP, from negative exams): %d", n_neg)
    logger.info("  candidates excluded (ambiguous multi-candidate positive exam): %d", n_excluded_ambiguous)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks-dir", default="segmentation/nnunet_output")
    parser.add_argument("--mapping-csv", default="segmentation/mapping.csv")
    parser.add_argument("--output-csv", default="segmentation/candidates.csv")
    args = parser.parse_args()

    rows = extract_candidates(args.masks_dir, args.mapping_csv)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exam_id", "component_id", "cz", "cy", "cx", "voxel_count",
        "true_volume_label", "n_components_in_exam", "patch_label", "source",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d candidates to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
