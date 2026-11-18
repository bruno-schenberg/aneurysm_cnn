"""
Rank all exams by noise severity and produce a visual atlas.

For each exam, samples every 4th Z slice, computes a noise score from the
outer ring of pixels (which should be air), then selects the single worst
slice as the representative. Produces a grid of exams ordered from noisiest
to cleanest.

Noise score per slice: ratio of outer-ring median to inner-region median.
High ratio = the "background" is bright relative to the brain = noisy.

Usage:
    python scripts/justification/noise_ranking.py
    python scripts/justification/noise_ranking.py --top 50
    python scripts/justification/noise_ranking.py --top 100 --cols 10
"""
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import config
from scripts.justification.common import open_file


def score_exam(path, ring_frac=0.1, z_step=4):
    """Compute per-slice noise scores, return (worst_score, worst_slice_idx).

    Noise score = median(outer_ring) / median(inner_region).
    A clean slice has score ~0 (dark ring, bright center).
    A noisy slice has score approaching 1 (ring as bright as center).
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    h, w, nz = data.shape
    margin_h = max(int(h * ring_frac), 1)
    margin_w = max(int(w * ring_frac), 1)

    worst_score = -1.0
    worst_z = 0

    for z in range(0, nz, z_step):
        slc = data[:, :, z]

        ring = np.concatenate([
            slc[:margin_h, :].ravel(),
            slc[-margin_h:, :].ravel(),
            slc[margin_h:-margin_h, :margin_w].ravel(),
            slc[margin_h:-margin_h, -margin_w:].ravel(),
        ])
        inner = slc[margin_h:-margin_h, margin_w:-margin_w].ravel()

        ring_med = np.median(ring[ring > 0]) if (ring > 0).any() else 0
        inner_med = np.median(inner[inner > 0]) if (inner > 0).any() else 1e-9

        score = ring_med / max(inner_med, 1e-9)

        if score > worst_score:
            worst_score = score
            worst_z = z

    return worst_score, worst_z


def get_worst_slice(path, z_idx):
    """Load and return a single Z slice + spacing."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    sp = img.header.get_zooms()
    slc = data[:, :, z_idx].T
    # Per-slice percentile windowing for display
    pos = slc[slc > 0]
    if len(pos) > 0:
        p1, p99 = np.percentile(pos, [1, 99])
        slc = np.clip(slc, p1, p99)
    return slc, sp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top", type=int, default=80,
                        help="Number of exams to show (default: 80)")
    parser.add_argument("--cols", type=int, default=10,
                        help="Grid columns (default: 10)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Discover all NIfTIs
    all_exams = []
    for cls_dir in sorted(config.NIFTI_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*.nii.gz")):
            all_exams.append((p, cls_dir.name))

    print(f"Scoring {len(all_exams)} exams ...", flush=True)

    results = []  # (score, z_idx, path, cls, exam_id)
    for i, (path, cls) in enumerate(all_exams):
        exam_id = path.stem.replace(".nii", "")
        score, z_idx = score_exam(path)
        results.append((score, z_idx, path, cls, exam_id))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(all_exams)}", flush=True)

    # Sort by noise score descending (noisiest first)
    results.sort(key=lambda x: x[0], reverse=True)

    # Save CSV
    out_dir = config.REPORTS_DIR / "noise_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "noise_ranking.csv"
    with open(csv_path, "w") as f:
        f.write("rank,exam_id,class,noise_score,worst_z\n")
        for rank, (score, z_idx, path, cls, eid) in enumerate(results, 1):
            f.write(f"{rank},{eid},{cls},{score:.4f},{z_idx}\n")
    print(f"Saved {csv_path} ({len(results)} exams)", flush=True)

    # Draw grid of top N noisiest
    n_show = min(args.top, len(results))
    n_cols = args.cols
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Noisiest {n_show} exams (outer-ring / inner-region ratio)",
                 color="white", fontsize=10, y=1.0)

    if n_rows == 1:
        axes = [axes]

    for i in range(n_show):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        score, z_idx, path, cls, eid = results[i]

        slc, sp = get_worst_slice(path, z_idx)
        ax.imshow(slc, cmap="gray", origin="upper", interpolation="bilinear")
        ax.set_facecolor("black")
        ax.axis("off")

        cls_label = "A" if cls == "1" else "C"
        ax.set_title(f"#{i+1} {eid} [{cls_label}]\n"
                     f"s={score:.2f} z={z_idx}",
                     fontsize=4.5, color="white", family="monospace", pad=2)

    # Hide unused cells
    for i in range(n_show, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_facecolor("black")
        axes[r][c].axis("off")

    plt.tight_layout(pad=0.3)
    out_path = args.out or str(out_dir / f"noise_ranking_top{n_show}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}")
    open_file(out_path)


if __name__ == "__main__":
    main()
