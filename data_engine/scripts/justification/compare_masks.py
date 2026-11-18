"""
Compare foreground masking + normalization strategies.

Shows how different preprocessing pipelines affect normalization quality,
especially for noisy exams like BP0971 where vol > 0 captures 99.7% of voxels.

Rows: one per strategy, each exam side by side.
All rows use fixed [0, 1] colormap for direct comparison.

Strategies tested:
  1. baseline: vol > 0 + percentile
  2. ring sub (median) + percentile
  3. ring sub (median) + histogram matching

Usage:
    python scripts/justification/compare_masks.py BP0971 BP0843
    python scripts/justification/compare_masks.py BP0971 BP0843 --slices 12
"""
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from scripts.justification.common import load_exam, open_file
from scripts.justification.compare_exams import get_histogram_template, norm_histogram


# -- Masking strategies --

def mask_gt_zero(vol):
    """Current baseline: vol > 0."""
    return vol > 0


def ring_denoise(vol, ring_frac=0.1):
    """Subtract per-slice noise floor estimated from outer ring (median).

    Returns (denoised_volume, mask_of_remaining_positive_voxels).
    """
    out = vol.copy()
    mask = np.zeros(vol.shape, dtype=bool)
    h, w = vol.shape[0], vol.shape[1]
    margin_h = max(int(h * ring_frac), 1)
    margin_w = max(int(w * ring_frac), 1)

    for z in range(vol.shape[2]):
        slc = vol[:, :, z]
        ring = np.concatenate([
            slc[:margin_h, :].ravel(),
            slc[-margin_h:, :].ravel(),
            slc[margin_h:-margin_h, :margin_w].ravel(),
            slc[margin_h:-margin_h, -margin_w:].ravel(),
        ])
        ring_pos = ring[ring > 0]
        noise_floor = np.median(ring_pos) if len(ring_pos) > 0 else 0
        denoised = np.clip(slc - noise_floor, 0, None)
        out[:, :, z] = denoised
        mask[:, :, z] = denoised > 0
    return out, mask


def norm_percentile_with_mask(vol, mask):
    """Percentile normalization using the provided mask instead of vol > 0."""
    fg = vol[mask]
    if len(fg) == 0:
        return vol.copy()
    p1, p99 = np.percentile(fg, [1, 99])
    out = np.clip(vol, 0, p99)
    out = np.where(mask, (out - p1) / max(p99 - p1, 1e-9), 0.0)
    return np.clip(out, 0, None).astype(np.float32)


def norm_histogram_with_mask(vol, mask, template):
    """Histogram matching using the provided mask instead of vol > 0."""
    from skimage.exposure import match_histograms
    fg = vol[mask]
    if len(fg) == 0:
        return vol.copy()
    matched_fg = match_histograms(fg, template, channel_axis=None)
    out = np.zeros_like(vol)
    out[mask] = matched_fg.astype(np.float32)
    return out


# -- Full pipelines: each takes raw vol and returns normalized [0,1]-ish output --

def pipeline_baseline(vol, **_kw):
    """Current: vol > 0 mask + percentile."""
    mask = vol > 0
    return norm_percentile_with_mask(vol, mask), mask

def pipeline_ring_percentile(vol, **_kw):
    """Ring denoise + percentile."""
    denoised, mask = ring_denoise(vol)
    return norm_percentile_with_mask(denoised, mask), mask

def pipeline_ring_histogram(vol, template=None, **_kw):
    """Ring denoise + histogram matching."""
    denoised, mask = ring_denoise(vol)
    return norm_histogram_with_mask(denoised, mask, template), mask


STRATEGIES = [
    ("baseline (vol>0 + pctl)", pipeline_baseline),
    ("ring sub + percentile", pipeline_ring_percentile),
    ("ring sub + histogram", pipeline_ring_histogram),
]


def make_comparison(exam_ids, n_slices, out_path):
    n_exams = len(exam_ids)
    template = get_histogram_template()

    # Load exams
    exams = {}
    for eid in exam_ids:
        data, sp, cls = load_exam(eid)
        exams[eid] = {"data": data, "sp": sp, "cls": cls}

    n_strategies = len(STRATEGIES)
    n_rows = n_strategies * n_exams
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(n_slices * 2.2 + 2, n_rows * 2.5))
    fig.patch.set_facecolor("black")

    if n_rows == 1:
        axes = [list(axes)] if n_slices > 1 else [[axes]]
    elif n_slices == 1:
        axes = [[ax] for ax in axes]

    # Compute vmin/vmax for histogram group across all exams
    hist_ranges = {}
    for strat_name, pipe_fn in STRATEGIES:
        if "histogram" not in strat_name:
            continue
        all_fg = []
        for eid in exam_ids:
            data = exams[eid]["data"]
            if data is None:
                continue
            vol, mask = pipe_fn(data, template=template)
            all_fg.append(vol[vol > 0])
        if all_fg:
            combined = np.concatenate(all_fg)
            hist_ranges[strat_name] = (0, np.percentile(combined, 99))

    row = 0
    for strat_name, pipe_fn in STRATEGIES:
        for eid in exam_ids:
            data = exams[eid]["data"]
            sp = exams[eid]["sp"]
            cls = exams[eid]["cls"]

            if data is None:
                for col in range(n_slices):
                    ax = axes[row][col]
                    ax.set_facecolor("black")
                    ax.text(0.5, 0.5, f"{eid}\nnot found",
                            ha="center", va="center", color="gray", fontsize=6,
                            transform=ax.transAxes)
                    ax.axis("off")
                row += 1
                continue

            vol, mask = pipe_fn(data, template=template)
            fg_pct = mask.sum() / data.size * 100

            if strat_name in hist_ranges:
                vmin, vmax = hist_ranges[strat_name]
            else:
                vmin, vmax = 0.0, 1.0

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
                            f"{eid} [{cls_label}]\n{strat_name}\n"
                            f"mask={fg_pct:.1f}%",
                            fontsize=5, color="white", family="monospace",
                            ha="right", va="center",
                            transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"z={si} ({z_mm:.0f}mm)", fontsize=5, color="white",
                                 family="monospace", pad=2)

            row += 1

    # Group separator lines
    for g in range(1, n_strategies):
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
                        help="Output PNG path")
    args = parser.parse_args()

    out_dir = config.REPORTS_DIR / "exam_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(out_dir / f"mask_comparison_{'_vs_'.join(args.exams)}.png")

    make_comparison(args.exams, args.slices, out_path)
    open_file(out_path)


if __name__ == "__main__":
    main()
