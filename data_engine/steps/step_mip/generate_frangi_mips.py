"""Generate Frangi-enhanced MIP datasets.

Two modes:
  --mode 3d: Compute 3D Frangi vesselness on NIfTI volumes (cuCIM GPU), then MIP
  --mode 2d: Compute 2D Frangi on existing axial MIP images (skimage CPU)

For 3D mode, supports GPU parallelism via --gpu flag (use CUDA_VISIBLE_DEVICES
externally or launch multiple processes).

Output datasets (under --output-root):
  mip_frangi_2ch/      — 2-channel [intensity_MIP, frangi_MIP], shape (2, H, W)
  mip_frangi_diff/     — 1-channel difference MIP, shape (H, W)
  mip_frangi2d_2ch/    — same but Frangi computed in 2D
  mip_frangi2d_diff/   — same but Frangi computed in 2D

Usage:
    # 3D Frangi on GPU (from NIfTI volumes):
    CUDA_VISIBLE_DEVICES=0 python -m steps.step_mip.generate_frangi_mips --mode 3d \\
        --source /path/to/c90z_384 --output-root /path/to/datasets

    # 2D Frangi on CPU (from existing MIP .npy):
    python -m steps.step_mip.generate_frangi_mips --mode 2d \\
        --source /path/to/mip_axial --output-root /path/to/datasets
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from medprep.mip import load_volume, resize_2d

from ._cli import (
    apply_shard,
    collect_class_files,
    ensure_variant_dirs,
    file_stem,
    print_variant_counts,
    save_variants,
    variant_outputs_exist,
)
from ._frangi import compute_vesselness_3d


# ---------------------------------------------------------------------------
# 3D Frangi (cuCIM on GPU)
# ---------------------------------------------------------------------------

def process_volume_3d(nifti_path: Path, target_size: int = 384) -> dict[str, np.ndarray]:
    """3D Frangi on GPU via cuCIM, then MIP.

    cpu_fallback=False: this mode is documented as GPU-only (see module
    docstring), so a missing cupy/cucim should raise immediately rather
    than silently fall back to a CPU path that's minutes-per-volume slow.
    """
    data = load_volume(nifti_path)
    target = (target_size, target_size)

    dmax = data.max()
    data_norm = data / dmax if dmax > 0 else data

    vesselness = compute_vesselness_3d(data_norm, cpu_fallback=False)

    # Axial MIPs (max along Z=axis 2)
    intensity_mip = np.max(data, axis=2)
    frangi_mip = np.max(vesselness, axis=2)

    intensity_mip = resize_2d(intensity_mip, target)
    frangi_mip = resize_2d(frangi_mip, target)

    # Normalize to [0, 1]
    imax = intensity_mip.max()
    if imax > 0:
        intensity_mip = intensity_mip / imax
    fmax = frangi_mip.max()
    if fmax > 0:
        frangi_mip = frangi_mip / fmax

    diff_mip = np.clip(intensity_mip - frangi_mip, 0, None).astype(np.float32)

    return {
        "2ch": np.stack([intensity_mip, frangi_mip], axis=0),
        "diff": diff_mip,
    }


# ---------------------------------------------------------------------------
# 2D Frangi (skimage on CPU)
# ---------------------------------------------------------------------------

def process_mip_2d(mip_path: Path) -> dict[str, np.ndarray]:
    """2D Frangi on CPU via skimage."""
    from skimage.filters import frangi as skimage_frangi

    intensity_mip = np.load(str(mip_path)).astype(np.float32)

    imax = intensity_mip.max()
    intensity_norm = intensity_mip / imax if imax > 0 else intensity_mip

    frangi_mip = skimage_frangi(
        intensity_norm, sigmas=range(1, 6), black_ridges=False,
    ).astype(np.float32)

    fmax = frangi_mip.max()
    if fmax > 0:
        frangi_mip = frangi_mip / fmax

    diff_mip = np.clip(intensity_norm - frangi_mip, 0, None).astype(np.float32)

    return {
        "2ch": np.stack([intensity_norm, frangi_mip], axis=0),
        "diff": diff_mip,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Frangi-enhanced MIP datasets")
    parser.add_argument("--mode", choices=["3d", "2d"], required=True)
    parser.add_argument("--source", required=True, help="Source dir with 0/ and 1/ subdirs")
    parser.add_argument("--output-root", required=True, help="Parent dir for output datasets")
    parser.add_argument("--target-size", type=int, default=384, help="Output MIP size (3D mode)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only a shard: 'K/N' (shard K of N, 0-indexed)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output_root)

    if not source_dir.exists():
        print(f"ERROR: Source not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    # Dataset variant names
    if args.mode == "3d":
        prefix = "mip_frangi"
        ext = (".nii", ".nii.gz")
    else:
        prefix = "mip_frangi2d"
        ext = (".npy",)

    variants = (f"{prefix}_2ch", f"{prefix}_diff")
    ensure_variant_dirs(output_root, variants)

    all_files = collect_class_files(source_dir, ext, verbose=False)
    print(f"Mode: {args.mode}, Total files: {len(all_files)}")

    # Sharding for GPU parallelism
    all_files = apply_shard(all_files, args.shard)

    total_generated = 0
    total_skipped = 0
    t_start = time.time()

    for i, (fpath, cls) in enumerate(all_files):
        stem = file_stem(fpath)

        if variant_outputs_exist(output_root, variants, cls, stem):
            total_skipped += 1
            continue

        if args.mode == "3d":
            results = process_volume_3d(fpath, args.target_size)
        else:
            results = process_mip_2d(fpath)

        save_variants(output_root, cls, stem, {
            f"{prefix}_2ch": results["2ch"],
            f"{prefix}_diff": results["diff"],
        })

        total_generated += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(all_files):
            elapsed = time.time() - t_start
            rate = total_generated / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(all_files)}] {stem} ({rate:.1f} files/s)")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s. Generated: {total_generated}, Skipped: {total_skipped}")
    print_variant_counts(output_root, variants)


if __name__ == "__main__":
    main()
