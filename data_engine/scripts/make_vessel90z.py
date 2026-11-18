"""vessel90z_256: the 3D counterpart of the final 2D window (mip_grid vessel90).

Same geometry as c90z_256 (256x256x128, 0.703125 mm iso, 180 mm XY, 90 mm Z), but the
90 mm Z window is each exam's vessel90 window from mip_grid/windows.csv (centred on the
slice with the most bright-vessel voxels) instead of the central 90 mm.

    .venv/bin/python scripts/make_vessel90z.py [--workers 8]
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from medprep.resample import pad_or_crop, resample_volume

NIFTI = Path("/mnt/data/nifti")
WINDOWS = Path("/mnt/data/datasets/mip_grid/windows.csv")
OUT = Path("/mnt/data/datasets/vessel90z_256")
SHAPE, SPACING = (256, 256, 128), (0.703125, 0.703125, 0.703125)


def load_windows() -> dict[str, tuple[float, float]]:
    win: dict[str, tuple[float, float]] = {}
    with open(WINDOWS) as f:
        for r in csv.DictReader(f):
            if r["window"] != "vessel90":
                continue
            b = (float(r["z_start_mm"]), float(r["z_end_mm"]))
            # windows.csv is appended per MIP run; all runs must agree on the bounds
            assert win.setdefault(r["exam"], b) == b, f"{r['exam']}: conflicting vessel90 bounds"
    return win


def one(args: tuple[Path, float, float]) -> str:
    path, z0, z1 = args
    out = OUT / path.parent.name / path.name
    if out.exists():
        return "skip"
    img = nib.load(str(path))
    # the MIP generator measures Z in mm from the inferior end of the LPS volume
    assert nib.aff2axcodes(img.affine) == ("L", "P", "S"), f"{path.name}: not LPS"
    data, aff = img.get_fdata(dtype=np.float32), img.affine.copy()
    sz = float(img.header.get_zooms()[2])
    a, b = int(round(z0 / sz)), int(round(z1 / sz))
    data = data[:, :, a:b]
    aff[:3, 3] += a * aff[:3, 2]  # keep world coordinates of the cropped slab
    data, aff = resample_volume(data, aff, SPACING)
    data, aff = pad_or_crop(data, aff, SHAPE)
    assert data.shape == SHAPE
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), aff), str(out))
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    win = load_windows()
    files = sorted(NIFTI.glob("[01]/*.nii.gz"))
    missing = [p.name for p in files if p.name[:-7] not in win]
    assert not missing, f"no vessel90 window for {missing[:5]}"
    jobs = [(p, *win[p.name[:-7]]) for p in files]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    print({s: res.count(s) for s in set(res)}, "->", OUT)


if __name__ == "__main__":
    main()
