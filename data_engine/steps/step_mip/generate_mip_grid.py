"""Unified MIP generator: mm-defined Z windows, identical for our data and Lausanne.

Replaces the per-experiment generate_*_mips.py scripts, which mixed physical
scales (top-N% of slices, Lausanne without the 180 mm XY crop).

Per volume:
  1. Load, reorient to LPS (our NIfTIs' native orientation; Z increases
     towards the vertex, so "top" = high Z index).
  2. Optional: 180 mm XY centroid crop (--crop-fov-mm, for Lausanne; ours are
     already cropped in step 1), N4 bias-field correction (--n4, in memory),
     dilated brain mask (--mask-dir).
  3. For each Z window: max-project the window at native spacing, then resize
     each view by its physical extent. Projecting before resizing keeps the
     native in-plane detail (down to 0.234 mm) for the 512/768 variants,
     which a 3D resample to 0.469 mm would throw away.

Windows (mm, measured on the reoriented volume):
  center90       90 mm centred on the Z extent (same anatomy as c90z_*)
  top60..top120  the top N mm
  vessel60/90    N mm centred on the argmax of the smoothed per-slice count
                 of >99.9th-percentile voxels (lands on the
                 carotid siphon, just below the CoW)
  pct90          top 90% of slices (legacy rule, now physically resized)
  full           the whole slab
A window that runs off the volume is shifted inside it and truncated only if
the volume is shorter than the window; windows.csv flags these as clipped.

Aspect:
  stretch  each view resized to size x size (the legacy behaviour)
  pad      each view scaled to fit its longer physical side into size,
           preserving mm aspect, then zero-padded to size x size

Output: <output-root>/<window>_<aspect>_<size>/<class>/<prefix><stem>.npy,
float16 (3, size, size) [axial, coronal, sagittal], plus <output-root>/windows.csv.

Usage:
    python -m steps.step_mip.generate_mip_grid \\
        --source /mnt/data/nifti --output-root /mnt/data/mip_grid --workers 8
    python -m steps.step_mip.generate_mip_grid \\
        --source /mnt/data/lausanne_nifti --output-root /mnt/data/mip_grid_lausanne \\
        --crop-fov-mm 180 --prefix laus_ --workers 8
"""
import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

from medprep.mip import resize_2d
from medprep.nifti_convert import centroid_crop_xy

from ._cli import apply_shard, collect_class_files, file_stem

WINDOWS = ("center90", "top60", "top75", "top90mm", "top120",
           "vessel60", "vessel90", "pct90", "full")
ASPECTS = ("stretch", "pad")
FOV_MM = 180.0
VESSEL_SMOOTH_MM = 10.0


def load_lps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI reoriented to LPS. Returns (data float32, spacing mm (3,))."""
    img = nib.load(str(path))
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine), nib.orientations.axcodes2ornt("LPS"))
    img = img.as_reoriented(ornt)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def n4_correct(data: np.ndarray, spacing) -> np.ndarray:
    """N4 bias-field correction, fitted at 1/4 resolution on an Otsu foreground mask."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(data.T)  # sitk is (z, y, x)
    img.SetSpacing([float(s) for s in spacing])
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(sitk.Shrink(img, [4] * 3), sitk.Shrink(mask, [4] * 3))
    log_bias = sitk.GetArrayFromImage(corrector.GetLogBiasFieldAsImage(img)).T
    return (data / np.exp(log_bias)).astype(np.float32)


def vessel_anchor_mm(data: np.ndarray, sz: float) -> float:
    """Z position (mm from the bottom) with the most bright-vessel voxels."""
    # 99.9th: the 99th peaks on neck fat + carotids; 99.9th lands on the siphon, below the CoW
    profile = (data > np.percentile(data, 99.9)).sum(axis=(0, 1)).astype(float)
    profile = gaussian_filter1d(profile, VESSEL_SMOOTH_MM / sz)
    return (int(np.argmax(profile)) + 0.5) * sz


def window_bounds(window: str, extent: float, anchor: float) -> tuple[float, float, bool]:
    """(z_start_mm, z_end_mm, clipped) for a window on a slab of `extent` mm."""
    if window == "full":
        return 0.0, extent, False
    if window == "pct90":
        return 0.1 * extent, extent, False
    width = float(re.search(r"\d+", window).group())
    if window.startswith("top"):
        start = extent - width
    elif window.startswith("center"):
        start = extent / 2 - width / 2
    else:  # vessel
        start = anchor - width / 2
    clipped = start < 0 or start + width > extent
    start = min(max(start, 0.0), max(extent - width, 0.0))
    return start, min(start + width, extent), clipped


def fit_view(mip: np.ndarray, extent_mm: tuple[float, float], size: int, aspect: str) -> np.ndarray:
    if aspect == "stretch":
        return resize_2d(mip, (size, size))
    scale = size / max(extent_mm)
    shape = tuple(max(1, round(e * scale)) for e in extent_mm)
    out = np.zeros((size, size), dtype=np.float32)
    r0, c0 = (size - shape[0]) // 2, (size - shape[1]) // 2
    out[r0:r0 + shape[0], c0:c0 + shape[1]] = resize_2d(mip, shape)
    return out


def process_volume(path: Path, mask_path: Path | None, args) -> tuple[dict, list[dict]]:
    """Returns ({variant: (3, S, S) float16}, windows.csv rows)."""
    data, affine = load_lps(path)
    if args.crop_fov_mm:
        data, affine, _, _ = centroid_crop_xy(data, affine, args.crop_fov_mm)
    if args.n4:
        data = n4_correct(data, nib.affines.voxel_sizes(affine))
    if mask_path is not None:
        mask, _ = load_lps(mask_path)
        spacing = nib.affines.voxel_sizes(affine)
        dist = distance_transform_edt(mask <= 0, sampling=spacing)
        data = np.where(dist <= args.mask_dilate_mm, data, 0.0).astype(np.float32)
    sx, sy, sz = nib.affines.voxel_sizes(affine)
    extent = data.shape[2] * sz
    anchor = vessel_anchor_mm(data, sz)

    arrays, rows = {}, []
    for window in args.windows:
        z0, z1, clipped = window_bounds(window, extent, anchor)
        i0, i1 = int(round(z0 / sz)), max(int(round(z1 / sz)), int(round(z0 / sz)) + 1)
        slab = data[:, :, i0:i1]
        wz = (i1 - i0) * sz
        views = [
            (slab.max(axis=2), (data.shape[0] * sx, data.shape[1] * sy)),  # axial (X, Y)
            (slab.max(axis=1), (data.shape[0] * sx, wz)),                  # coronal (X, Z)
            (slab.max(axis=0), (data.shape[1] * sy, wz)),                  # sagittal (Y, Z)
        ]
        for aspect in args.aspects:
            for size in args.sizes:
                arrays[f"{window}_{aspect}_{size}"] = np.stack(
                    [fit_view(m, e, size, aspect) for m, e in views]).astype(np.float16)
        rows.append({"window": window, "z_start_mm": round(z0, 1), "z_end_mm": round(z1, 1),
                     "extent_mm": round(extent, 1), "clipped": clipped})
    return arrays, rows


def _worker(path: Path, cls: str, args) -> list[dict]:
    stem = args.prefix + file_stem(path)
    mask_path = None
    if args.mask_dir:
        mask_path = Path(args.mask_dir) / cls / path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"no mask {mask_path}")
    arrays, rows = process_volume(path, mask_path, args)
    for variant, arr in arrays.items():
        out = Path(args.output_root) / variant / cls / f"{stem}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, arr)
    return [{"exam": stem, "class": cls, **r} for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, help="NIfTI dir with 0/ and 1/")
    p.add_argument("--output-root", required=True)
    p.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=WINDOWS)
    p.add_argument("--aspects", nargs="+", default=list(ASPECTS), choices=ASPECTS)
    p.add_argument("--sizes", nargs="+", type=int, default=[384])
    p.add_argument("--crop-fov-mm", type=float, default=None,
                   help="180 for Lausanne (ours are cropped in step 1)")
    p.add_argument("--n4", action="store_true", help="N4 bias-field correction first")
    p.add_argument("--mask-dir", default=None, help="brain masks laid out like --source (<class>/<file>)")
    p.add_argument("--mask-dilate-mm", type=float, default=10.0)
    p.add_argument("--prefix", default="", help="output stem prefix, e.g. laus_")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", default=None, help="K/N")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"ERROR: source not found: {source}")
    if args.mask_dir and args.crop_fov_mm:
        sys.exit("ERROR: --mask-dir needs pre-cropped volumes; drop --crop-fov-mm")
    out_root = Path(args.output_root)
    variants = [f"{w}_{a}_{s}" for w in args.windows for a in args.aspects for s in args.sizes]

    files = apply_shard(collect_class_files(source, (".nii", ".nii.gz")), args.shard)
    todo = [(f, c) for f, c in files if args.overwrite or not all(
        (out_root / v / c / f"{args.prefix}{file_stem(f)}.npy").exists() for v in variants)]
    print(f"{len(variants)} variants, {len(todo)}/{len(files)} volumes to do")

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "windows.csv"
    new_file = not csv_path.exists()
    errors = 0
    with open(csv_path, "a", newline="") as fh, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(fh, ["exam", "class", "window", "z_start_mm", "z_end_mm",
                                     "extent_mm", "clipped"])
        if new_file:
            writer.writeheader()
        futures = {pool.submit(_worker, f, c, args): f for f, c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                writer.writerows(fut.result())
                fh.flush()
            except Exception as e:  # keep going; report at the end
                errors += 1
                print(f"  ERROR {futures[fut].name}: {e}")
            if i % 50 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}]")
    print(f"Done. Errors: {errors}. Window bounds -> {csv_path}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
