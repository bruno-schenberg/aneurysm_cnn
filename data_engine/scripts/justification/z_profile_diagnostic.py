"""
Z-axis profile diagnostic for two-sided trim.

Plots multiple per-slice metrics on ~10 representative exams to find which
metric gives a clean skull-base / vertex transition for Z cropping.

Metrics plotted per Z slice:
  1. Nonzero voxel count (baseline — expected to be flat on noisy exams)
  2. Voxels above volume p20 (mild threshold)
  3. Voxels above volume p50 (aggressive threshold)
  4. Mean intensity of the slice
  5. Inner-vs-outer ring: mean(inner 60%) - mean(outer ring)

Usage:
    cd data_engine
    python scripts/justification/z_profile_diagnostic.py
    python scripts/justification/z_profile_diagnostic.py --exams BP0971 BP0843 BP0310
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import config

# Representative exams spanning the dataset range
DEFAULT_EXAMS = [
    # (exam_id, class, reason)
    ("BP0310", 0, "noisiest (DR=6.9, 99% nz)"),
    ("BP0461", 1, "noisy aneurysm (DR=8.3, 100% nz)"),
    ("BP0150", 0, "lowest nonzero (78%, DR=223)"),
    ("BP0843", 1, "cleanest aneurysm (DR=436, 93% nz)"),
    ("BP0971", 1, "dental artifact (DR=21.8, 99.7% nz)"),
    ("BP0542", 0, "shortest Z (100.8mm)"),
    ("BP0739", 1, "longest Z (168mm, aneurysm)"),
    ("BP0032", 1, "0.75mm Z spacing aneurysm"),
    ("BP0300", 0, "highest DR (900.6, 100% nz)"),
    ("BP0644", 0, "long Z clean (DR=211, 94.5% nz)"),
]


def compute_z_profiles(data: np.ndarray) -> dict:
    """Compute multiple per-slice metrics for the Z axis."""
    nx, ny, nz = data.shape

    # Precompute volume-level thresholds
    pos = data[data > 0]
    if len(pos) > 0:
        p20 = np.percentile(pos, 20)
        p50 = np.percentile(pos, 50)
    else:
        p20 = p50 = 0

    # Inner/outer ring masks (same for all slices since XY shape is constant)
    yy, xx = np.ogrid[:nx, :ny]
    cx, cy = nx / 2.0, ny / 2.0
    # Radius normalized so r=1 touches the nearest edge
    r_max = min(cx, cy)
    r2 = ((yy - cx) / r_max) ** 2 + ((xx - cy) / r_max) ** 2
    inner_mask = r2 <= 0.36  # r <= 0.6 → inner 60%
    outer_mask = r2 >= 0.64  # r >= 0.8 → outer ring (80-100%)

    profiles = {
        "nonzero_count": np.zeros(nz),
        "above_p20": np.zeros(nz),
        "above_p50": np.zeros(nz),
        "mean_intensity": np.zeros(nz),
        "inner_minus_outer": np.zeros(nz),
    }

    for z in range(nz):
        slc = data[:, :, z]
        profiles["nonzero_count"][z] = (slc > 0).sum()
        profiles["above_p20"][z] = (slc > p20).sum()
        profiles["above_p50"][z] = (slc > p50).sum()
        profiles["mean_intensity"][z] = slc.mean()

        inner_vals = slc[inner_mask]
        outer_vals = slc[outer_mask]
        inner_mean = inner_vals.mean() if inner_vals.size > 0 else 0
        outer_mean = outer_vals.mean() if outer_vals.size > 0 else 0
        profiles["inner_minus_outer"][z] = inner_mean - outer_mean

    return profiles


def plot_profiles(exams_data: list, out_path: str):
    """Plot Z profiles for all exams, one row per exam, one column per metric."""
    metric_names = [
        ("nonzero_count", "Nonzero count"),
        ("above_p20", "Above p20"),
        ("above_p50", "Above p50"),
        ("mean_intensity", "Mean intensity"),
        ("inner_minus_outer", "Inner − Outer"),
    ]
    n_metrics = len(metric_names)
    n_exams = len(exams_data)

    fig, axes = plt.subplots(
        n_exams, n_metrics,
        figsize=(n_metrics * 3.5, n_exams * 2.0),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row, (exam_id, cls, reason, profiles, z_mm) in enumerate(exams_data):
        nz = len(profiles["nonzero_count"])
        z_axis = np.linspace(0, z_mm, nz)

        for col, (key, label) in enumerate(metric_names):
            ax = axes[row, col]
            values = profiles[key]

            # Normalize to [0, 1] for visual comparison
            vmax = values.max()
            if vmax > 0:
                normed = values / vmax
            else:
                normed = values

            ax.fill_between(z_axis, normed, alpha=0.3, color="steelblue")
            ax.plot(z_axis, normed, color="steelblue", linewidth=0.8)

            # Mark 20% threshold of peak
            threshold = 0.2
            ax.axhline(threshold, color="red", linewidth=0.6, linestyle="--", alpha=0.7)

            # Find trim points (first/last slice above 20% of peak)
            above = normed > threshold
            if above.any():
                first = np.argmax(above)
                last = nz - 1 - np.argmax(above[::-1])
                trim_lo_mm = z_axis[first]
                trim_hi_mm = z_axis[last]
                kept_mm = trim_hi_mm - trim_lo_mm
                ax.axvline(trim_lo_mm, color="green", linewidth=0.8, linestyle="-", alpha=0.8)
                ax.axvline(trim_hi_mm, color="green", linewidth=0.8, linestyle="-", alpha=0.8)
                ax.text(
                    0.98, 0.95,
                    f"{kept_mm:.0f}mm",
                    transform=ax.transAxes, fontsize=6,
                    ha="right", va="top", color="green",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="green", alpha=0.7),
                )

            ax.set_xlim(0, z_mm)
            ax.set_ylim(-0.05, 1.15)

            if row == 0:
                ax.set_title(label, fontsize=8, fontweight="bold")
            if col == 0:
                cls_label = "A" if cls == 1 else "C"
                ax.set_ylabel(
                    f"{exam_id} [{cls_label}]\n{reason}",
                    fontsize=5.5, rotation=0, labelpad=80, va="center",
                )
            if row == n_exams - 1:
                ax.set_xlabel("Z position (mm)", fontsize=6)

            ax.tick_params(labelsize=5)

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exams", nargs="+", default=None,
                        help="Specific exam IDs to plot (default: 10 representative)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: reports/z_crop/z_profiles.png)")
    args = parser.parse_args()

    out_dir = config.REPORTS_DIR / "z_crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(out_dir / "z_profiles.png")

    # Build exam list
    if args.exams:
        exam_list = [(eid, None, "user-specified") for eid in args.exams]
    else:
        exam_list = DEFAULT_EXAMS

    # Resolve classes for user-specified exams
    resolved = []
    for exam_id, cls, reason in exam_list:
        if cls is not None:
            nifti_path = config.NIFTI_ROOT / str(cls) / f"{exam_id}.nii.gz"
        else:
            # Try both classes
            for c in [0, 1]:
                p = config.NIFTI_ROOT / str(c) / f"{exam_id}.nii.gz"
                if p.exists():
                    nifti_path = p
                    cls = c
                    break
            else:
                print(f"  WARNING: {exam_id} not found, skipping")
                continue

        if not nifti_path.exists():
            print(f"  WARNING: {nifti_path} not found, skipping")
            continue
        resolved.append((exam_id, cls, reason, str(nifti_path)))

    # Load and compute profiles
    exams_data = []
    for exam_id, cls, reason, path in resolved:
        print(f"  Loading {exam_id}...")
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        sp = img.header.get_zooms()
        z_mm = data.shape[2] * float(sp[2])

        profiles = compute_z_profiles(data)
        exams_data.append((exam_id, cls, reason, profiles, z_mm))

    if not exams_data:
        print("No exams loaded.")
        return

    plot_profiles(exams_data, out_path)
    print(f"\nDone — {len(exams_data)} exams plotted.")


if __name__ == "__main__":
    main()
