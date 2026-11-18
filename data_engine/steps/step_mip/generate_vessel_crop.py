"""Vessel-aware 3D cropping (Phase 7C).

Reduces the search space for 3D classifiers by cropping each volume around
its own vascular tree instead of a generic anatomical center-crop: computes
3D Frangi vesselness, thresholds it, finds the vessel-tree bounding box
(with a voxel margin), and center-crops/pads to a fixed output shape around
that box. Aneurysms occur on vessels, so a tight vessel-tree crop should
carry them along while discarding most of the skull/soft-tissue volume a
plain center-crop keeps.

This is the simpler alternative to full Circle-of-Willis atlas registration —
no external MNI template or registration library needed, and it's testable end-to-end without one.
Atlas registration may still be worth trying later if this isn't tight/
consistent enough once run against real data.

Frangi runs on GPU via cuCIM by default (matches generate_frangi_mips.py's
3D mode — full-volume 3D vesselness on CPU is impractically slow at our
scale: minutes per volume x hundreds of volumes). Falls back to skimage
(CPU) when cuCIM is unavailable, which is what this was validated with
locally on synthetic data (no GPU here) — real runs on the cluster should
hit the GPU path.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m steps.step_mip.generate_vessel_crop \\
        --source /path/to/c90z_384 --output /path/to/vessel_crop_256 \\
        --target-shape 256 256 128
"""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from medprep.resample import pad_or_crop

from ._cli import apply_shard, collect_class_files, file_stem
from ._frangi import compute_vesselness_3d


def _vessel_bbox(vesselness: np.ndarray, percentile: float, margin: int) -> tuple[slice, slice, slice]:
    """Bounding box of the top-`percentile` vesselness voxels, expanded by `margin`."""
    threshold = np.percentile(vesselness, percentile)
    mask = vesselness >= threshold
    if not mask.any():
        # Degenerate case (e.g. flat/empty volume) — no crop, keep full extent.
        return tuple(slice(0, s) for s in vesselness.shape)

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    slices = []
    for dim in range(3):
        lo = max(int(mins[dim]) - margin, 0)
        hi = min(int(maxs[dim]) + margin, vesselness.shape[dim])
        slices.append(slice(lo, hi))
    return tuple(slices)


def process_volume(
    nifti_path: Path, target_shape: tuple[int, int, int],
    percentile: float, margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(nifti_path))
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    dmax = data.max()
    data_norm = data / dmax if dmax > 0 else data

    vesselness = compute_vesselness_3d(data_norm)
    bbox = _vessel_bbox(vesselness, percentile, margin)

    cropped = data[bbox]
    cropped_affine = affine.copy()
    cropped_affine[:3, 3] += affine[:3, :3] @ np.array([s.start for s in bbox])

    return pad_or_crop(cropped, cropped_affine, target_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Source dir with 0/ and 1/ subdirs")
    parser.add_argument("--output", required=True, help="Output dataset dir")
    parser.add_argument("--target-shape", type=int, nargs=3, default=[256, 256, 128])
    parser.add_argument(
        "--vessel-percentile", type=float, default=98.0,
        help="Vesselness percentile threshold for the vessel mask (default: top 2%%)",
    )
    parser.add_argument("--margin", type=int, default=20, help="Voxel margin around the vessel bbox")
    parser.add_argument("--shard", type=str, default=None, help="'K/N' (shard K of N, 0-indexed)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    target_shape = tuple(args.target_shape)

    if not source_dir.exists():
        print(f"ERROR: Source not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    for cls in ("0", "1"):
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    all_files = collect_class_files(source_dir, (".nii", ".nii.gz"), verbose=False)
    print(f"Total files: {len(all_files)}, target shape: {target_shape}")

    all_files = apply_shard(all_files, args.shard, show_total=False)

    n_done, n_skipped = 0, 0
    t_start = time.time()
    for i, (fpath, cls) in enumerate(all_files):
        stem = file_stem(fpath)
        out_path = output_dir / cls / f"{stem}.nii.gz"
        if out_path.exists():
            n_skipped += 1
            continue

        final, final_affine = process_volume(fpath, target_shape, args.vessel_percentile, args.margin)
        nib.save(nib.Nifti1Image(final, final_affine), str(out_path))

        n_done += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(all_files):
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [{i + 1}/{len(all_files)}] {stem} ({rate:.2f} files/s)")

    print(f"\nDone in {time.time() - t_start:.0f}s. Generated: {n_done}, Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
