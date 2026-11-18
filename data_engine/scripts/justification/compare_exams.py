"""
Multi-slice comparison of exams with normalization methods side by side.

# Reproduce the baseline normalization comparison (vol > 0 mask):
#   python scripts/justification/compare_exams.py BP0971 BP0843

Shows axial slices per exam in groups: raw, then each normalization method.
Row order: exam1 raw, exam2 raw, exam1 percentile, exam2 percentile,
           exam1 zscore, exam2 zscore, exam1 histogram, exam2 histogram.

All images within a normalization group share the same fixed colormap range
so visual brightness is directly comparable across exams.

Usage:
    python scripts/justification/compare_exams.py BP0971 BP0843
    python scripts/justification/compare_exams.py BP0971 BP0843 --slices 12
    python scripts/justification/compare_exams.py BP0971 BP0843 --out /tmp/comparison.png
    python scripts/justification/compare_exams.py BP0971 BP0843 --raw-only
"""
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import config
from scripts.justification.common import load_exam, norm_percentile, open_file


HISTOGRAM_TEMPLATE_PATH = config.DATASET_CSV.parent / "histogram_template.npy"
N_BINS = 1024


# -- Normalization functions (operate on whole volumes) --

def norm_zscore(vol):
    """Z-score normalization of foreground. Background stays zero."""
    fg = vol[vol > 0]
    if len(fg) == 0:
        return vol.copy()
    mu, sigma = fg.mean(), max(fg.std(), 1e-9)
    return np.where(vol > 0, (vol - mu) / sigma, 0.0).astype(np.float32)


def norm_histogram(vol, template):
    """Match foreground histogram to a quantile template."""
    from skimage.exposure import match_histograms
    mask = vol > 0
    fg = vol[mask]
    if len(fg) == 0:
        return vol.copy()
    matched_fg = match_histograms(fg, template, channel_axis=None)
    out = np.zeros_like(vol)
    out[mask] = matched_fg.astype(np.float32)
    return out


def build_histogram_template_from_niftis(n_sample=50, seed=42):
    """Build a histogram template from a random sample of NIfTI exams."""
    all_paths = []
    for cls_dir in sorted(config.NIFTI_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        all_paths.extend(sorted(cls_dir.glob("*.nii.gz")))

    rng = np.random.default_rng(seed)
    rng.shuffle(all_paths)
    sample = all_paths[:n_sample]

    print(f"Building histogram template from {len(sample)}/{len(all_paths)} exams ...", flush=True)
    quantiles_q = np.linspace(0, 1, N_BINS)
    templates = []
    for i, p in enumerate(sample):
        vol = nib.load(str(p)).get_fdata(dtype=np.float32)
        fg = vol[vol > 0]
        if len(fg) == 0:
            continue
        templates.append(np.quantile(fg, quantiles_q))
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(sample)}", flush=True)
    if not templates:
        raise RuntimeError(f"No NIfTI volumes found in {config.NIFTI_ROOT}")
    avg = np.mean(templates, axis=0).astype(np.float32)
    print(f"Template built from {len(templates)} volumes ({N_BINS} bins)", flush=True)
    return avg


def get_histogram_template():
    """Load existing template or build from all NIfTIs."""
    if HISTOGRAM_TEMPLATE_PATH.exists():
        print(f"Loading histogram template from {HISTOGRAM_TEMPLATE_PATH}")
        return np.load(HISTOGRAM_TEMPLATE_PATH).astype(np.float32)
    template = build_histogram_template_from_niftis()
    HISTOGRAM_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(HISTOGRAM_TEMPLATE_PATH, template)
    print(f"Saved template to {HISTOGRAM_TEMPLATE_PATH}")
    return template


def make_comparison(exam_ids, n_slices, out_path, raw_only=False):
    n_exams = len(exam_ids)

    # Load all exams
    exams = {}
    for eid in exam_ids:
        data, sp, cls = load_exam(eid)
        exams[eid] = {"data": data, "sp": sp, "cls": cls}

    # Define normalization groups
    if raw_only:
        groups = [("raw", None)]
    else:
        template = get_histogram_template()
        groups = [
            ("raw", None),
            ("percentile [0,1]", "percentile"),
            ("z-score", "zscore"),
            ("histogram match", "histogram"),
        ]

    n_groups = len(groups)
    n_rows = n_groups * n_exams
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(n_slices * 2.2 + 2, n_rows * 2.5))
    fig.patch.set_facecolor("black")

    if n_rows == 1:
        axes = [list(axes)] if n_slices > 1 else [[axes]]
    elif n_slices == 1:
        axes = [[ax] for ax in axes]

    # Precompute normalized volumes and global ranges per group
    norm_vols = {}  # (group_name, exam_id) -> volume
    for group_name, norm_type in groups:
        for eid in exam_ids:
            data = exams[eid]["data"]
            if data is None:
                norm_vols[(group_name, eid)] = None
                continue
            if norm_type is None:
                norm_vols[(group_name, eid)] = data
            elif norm_type == "percentile":
                norm_vols[(group_name, eid)] = norm_percentile(data)
            elif norm_type == "zscore":
                norm_vols[(group_name, eid)] = norm_zscore(data)
            elif norm_type == "histogram":
                norm_vols[(group_name, eid)] = norm_histogram(data, template)

    # Fixed colormap ranges per group (computed across all exams in that group)
    group_ranges = {}
    for group_name, norm_type in groups:
        if norm_type == "percentile":
            group_ranges[group_name] = (0.0, 1.0)
        elif norm_type == "zscore":
            # Gather all zscore values to find a good fixed range
            all_fg = []
            for eid in exam_ids:
                v = norm_vols[(group_name, eid)]
                if v is not None:
                    all_fg.append(v[v != 0])
            if all_fg:
                combined = np.concatenate(all_fg)
                group_ranges[group_name] = (np.percentile(combined, 1), np.percentile(combined, 99))
            else:
                group_ranges[group_name] = (-2, 4)
        elif norm_type == "histogram":
            all_fg = []
            for eid in exam_ids:
                v = norm_vols[(group_name, eid)]
                if v is not None:
                    all_fg.append(v[v != 0])
            if all_fg:
                combined = np.concatenate(all_fg)
                group_ranges[group_name] = (np.percentile(combined, 1), np.percentile(combined, 99))
            else:
                group_ranges[group_name] = (0, 1)
        else:
            # Raw: use shared p1/p99 across all exams
            all_fg = []
            for eid in exam_ids:
                v = norm_vols[(group_name, eid)]
                if v is not None:
                    pos = v[v > 0]
                    if len(pos) > 0:
                        all_fg.append(pos)
            if all_fg:
                combined = np.concatenate(all_fg)
                group_ranges[group_name] = (np.percentile(combined, 1), np.percentile(combined, 99))
            else:
                group_ranges[group_name] = (0, 1)

    # Draw
    row = 0
    for group_name, norm_type in groups:
        vmin, vmax = group_ranges[group_name]
        for eid in exam_ids:
            vol = norm_vols[(group_name, eid)]
            sp = exams[eid]["sp"]
            cls = exams[eid]["cls"]

            if vol is None:
                for col in range(n_slices):
                    ax = axes[row][col]
                    ax.set_facecolor("black")
                    ax.text(0.5, 0.5, f"{eid}\nnot found",
                            ha="center", va="center", color="gray", fontsize=6,
                            transform=ax.transAxes)
                    ax.axis("off")
                row += 1
                continue

            nz = vol.shape[2]
            slice_indices = np.linspace(int(nz * 0.1), int(nz * 0.9), n_slices, dtype=int)

            for col, si in enumerate(slice_indices):
                ax = axes[row][col]
                slc = vol[:, :, si].T
                ax.imshow(slc, cmap="gray", origin="upper",
                          interpolation="bilinear", vmin=vmin, vmax=vmax)
                ax.set_facecolor("black")
                ax.axis("off")

                z_mm = si * sp[2]
                if col == 0:
                    cls_label = "aneurysm" if cls == "1" else "control"
                    ax.text(-0.02, 0.5,
                            f"{eid} [{cls_label}]\n{group_name}\n"
                            f"[{vmin:.2f}, {vmax:.2f}]",
                            fontsize=5, color="white", family="monospace",
                            ha="right", va="center",
                            transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"z={si} ({z_mm:.0f}mm)", fontsize=5, color="white",
                                 family="monospace", pad=2)

            row += 1

    # Add group separator lines
    if n_groups > 1:
        for g in range(1, n_groups):
            y = 1.0 - (g * n_exams) / n_rows
            fig.add_artist(plt.Line2D([0.02, 0.98], [y, y],
                           transform=fig.transFigure, color="yellow",
                           linewidth=0.8, alpha=0.6))

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("exams", nargs="+", help="Exam IDs to compare")
    parser.add_argument("--slices", type=int, default=10,
                        help="Number of Z slices per exam (default 10)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: reports/exam_comparisons/<IDs>.png)")
    parser.add_argument("--raw-only", action="store_true",
                        help="Only show raw (unnormalized) rows")
    args = parser.parse_args()

    out_dir = config.REPORTS_DIR / "exam_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(out_dir / f"norm_comparison_{'_vs_'.join(args.exams)}.png")

    make_comparison(args.exams, args.slices, out_path, raw_only=args.raw_only)
    open_file(out_path)


if __name__ == "__main__":
    main()
