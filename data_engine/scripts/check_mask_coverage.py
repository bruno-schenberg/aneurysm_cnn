"""Does a brain mask cut aneurysms or the ICAs? (tasks 2.3b, no visual reading)

1. Aneurysm coverage: positive exams with exactly one nnU-Net candidate
   (patch_label=1) give an approximate aneurysm position. Reports the % of
   them inside the mask after dilating it by 0/5/10/15 mm. Pick the smallest
   dilation that keeps 100% as generate_mip_grid --mask-dilate-mm.
2. Vessel coverage: % of bright-vessel voxels (>99.5th pct) inside the mask
   at that dilation; flags exams below 95%.

Usage:
    python scripts/check_mask_coverage.py --masks /mnt/data/masks_synthstrip
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
CANDIDATES = REPO / "segmentation" / "evaluation_c90z_384" / "candidates.csv"
DILATIONS_MM = (0, 5, 10, 15)


def outside_distance_mm(mask_img) -> np.ndarray:
    """Per-voxel distance (mm) to the mask; 0 inside."""
    mask = np.asarray(mask_img.dataobj) > 0
    return distance_transform_edt(~mask, sampling=nib.affines.voxel_sizes(mask_img.affine))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--masks", required=True, help="<masks>/<class>/<exam>.nii.gz, native NIfTI grid")
    p.add_argument("--nifti", default="/mnt/data/nifti")
    p.add_argument("--c90z", default="/mnt/data/datasets/c90z_384")
    args = p.parse_args()

    cand = pd.read_csv(CANDIDATES)
    single = cand[(cand.n_components_in_exam == 1) & (cand.patch_label == 1)]
    dists = []
    for r in single.itertuples():
        mask_img = nib.load(f"{args.masks}/1/{r.exam_id}.nii.gz")
        # despite the names, (cz, cy, cx) are c90z_384 NIfTI voxel (i, j, k): checked, the
        # candidates land on >99.8th-percentile voxels this way and off-volume the other way
        world = nib.load(f"{args.c90z}/1/{r.exam_id}.nii.gz").affine @ [r.cz, r.cy, r.cx, 1]
        i, j, k = np.round(np.linalg.inv(mask_img.affine) @ world)[:3].astype(int)
        dist = outside_distance_mm(mask_img)
        inside = all(0 <= v < s for v, s in zip((i, j, k), dist.shape))
        dists.append(dist[i, j, k] if inside else np.inf)
    dists = np.array(dists)
    print(f"aneurysm coverage ({len(dists)} single-candidate positives):")
    for d in DILATIONS_MM:
        print(f"  dilation {d:2d} mm: {np.mean(dists <= d):.1%} inside")
    print(f"  max distance outside the mask: {dists.max():.1f} mm")
    keep = next((d for d in DILATIONS_MM if np.all(dists <= d)), None)
    print(f"-> smallest dilation keeping 100%: {keep} mm" if keep is not None
          else "-> no tested dilation keeps 100%; masking is unsafe for 2D-1b")

    dil = keep if keep is not None else max(DILATIONS_MM)
    rows = []
    for f in sorted(Path(args.nifti).glob("*/*.nii.gz")):
        mask_img = nib.load(f"{args.masks}/{f.parent.name}/{f.name}")
        vol = np.asarray(nib.load(f).dataobj, dtype=np.float32)
        vessels = vol > np.percentile(vol, 99.5)
        rows.append((f.name[:-7], float(np.mean(outside_distance_mm(mask_img)[vessels] <= dil))))
    cov = pd.DataFrame(rows, columns=["exam", "vessel_coverage"])
    out = Path(args.masks) / f"vessel_coverage_dil{dil}mm.csv"
    cov.to_csv(out, index=False)
    low = cov[cov.vessel_coverage < 0.95]
    print(f"vessel coverage at {dil} mm: median {cov.vessel_coverage.median():.1%}, "
          f"{len(low)} exams < 95% -> {out}")


if __name__ == "__main__":
    main()
