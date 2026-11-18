"""
Compare intensity distributions across scanners before/after normalization.

Design-justification script: shows whether normalization strategies make
intensity distributions converge across different scanners/field strengths.

Each row = one normalization method (Original, Percentile, Z-score, Histogram matching).
Left: overlaid histograms of foreground voxels from Skyra vs each scanner.
Right: mid-slice images side by side.

Usage:
    python training_engine/scripts/diagnostics/view_normalization.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import config
from src.data import PercentileNormalized, ZScoreNormalized, HistogramMatched

DATASET_DIR = config.DATASET_PATHS["iso_256"]
TEMPLATE_PATH = Path(config.DEFAULT_CONFIG["histogram_template"])

# One fixed Skyra 3T exam vs. one exam each from the other scanner/field-
# strength combinations in the dataset — picked by hand to cover that
# spread, not a random or representative sample.
EXAM_PAIRS = [
    ("BP0001", "skyra 3T", "BP0030", "signa_hdxt 1.5T"),
    ("BP0001", "skyra 3T", "BP0029", "optima 1.5T"),
    ("BP0001", "skyra 3T", "BP0028", "signa_hdxt 3T"),
]

COLORS_A = "#4488ff"
COLORS_B = "#ff6644"


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def find_volume(exam_id: str) -> Path | None:
    for cls in ("0", "1"):
        p = DATASET_DIR / cls / f"{exam_id}.nii.gz"
        if p.exists():
            return p
    return None


def load_vol(exam_id: str) -> np.ndarray | None:
    path = find_volume(exam_id)
    if path is None:
        return None
    return nib.load(path).get_fdata(dtype=np.float32)


# ---------------------------------------------------------------------------
# Normalization wrappers (use training_engine transforms on raw volumes)
# ---------------------------------------------------------------------------

def _apply_transform(transform, vol: np.ndarray) -> np.ndarray:
    """Wrap a volume in a dict, apply a MapTransform, unwrap."""
    result = transform({"image": vol})
    return result["image"]


def build_norm_functions() -> list[tuple[str, callable]]:
    key = ["image"]
    percentile = PercentileNormalized(keys=key)
    zscore = ZScoreNormalized(keys=key)

    methods = [
        ("Original", lambda v: v),
        ("Percentile [p1,p99]→[0,1]", lambda v: _apply_transform(percentile, v)),
        ("Z-score", lambda v: _apply_transform(zscore, v)),
    ]

    if TEMPLATE_PATH.exists():
        hist = HistogramMatched(keys=key, template_path=str(TEMPLATE_PATH))
        methods.append(("Histogram matching", lambda v: _apply_transform(hist, v)))
    else:
        print(f"WARNING: {TEMPLATE_PATH} not found — histogram matching row skipped.")
        print("  Run: python data_engine/scripts/build_histogram_template.py")

    return methods


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def main() -> None:
    methods = build_norm_functions()

    n_pairs = len(EXAM_PAIRS)
    n_methods = len(methods)
    cols_per_pair = 3
    n_cols = n_pairs * cols_per_pair

    fig, axes = plt.subplots(
        n_methods, n_cols,
        figsize=(n_cols * 3.5, n_methods * 3.5),
        squeeze=False,
    )
    fig.patch.set_facecolor("black")
    fig.suptitle(
        "Intensity distributions: do they converge after normalization?",
        fontsize=13, color="white", family="monospace", y=1.01,
    )

    for r, (method_name, norm_fn) in enumerate(methods):
        for p, (exam_a, label_a, exam_b, label_b) in enumerate(EXAM_PAIRS):
            vol_a_raw = load_vol(exam_a)
            vol_b_raw = load_vol(exam_b)
            normed_a = norm_fn(vol_a_raw) if vol_a_raw is not None else None
            normed_b = norm_fn(vol_b_raw) if vol_b_raw is not None else None

            if normed_a is None or normed_b is None:
                base_col = p * cols_per_pair
                for c in range(cols_per_pair):
                    ax = axes[r, base_col + c]
                    ax.set_facecolor("black")
                    ax.axis("off")
                    ax.text(0.5, 0.5, "volume\nnot found",
                            ha="center", va="center", color="red",
                            fontsize=7, transform=ax.transAxes)
                continue

            fg_a = normed_a[vol_a_raw > 0]
            fg_b = normed_b[vol_b_raw > 0]

            base_col = p * cols_per_pair

            # -- Histogram --
            ax_hist = axes[r, base_col]
            ax_hist.set_facecolor("#111111")
            all_fg = np.concatenate([fg_a, fg_b])
            bins = np.linspace(min(fg_a.min(), fg_b.min()),
                               np.percentile(all_fg, 99.5),
                               150)
            ax_hist.hist(fg_a, bins=bins, alpha=0.6, color=COLORS_A, density=True,
                         label=f"{exam_a} ({label_a})")
            ax_hist.hist(fg_b, bins=bins, alpha=0.6, color=COLORS_B, density=True,
                         label=f"{exam_b} ({label_b})")
            ax_hist.legend(fontsize=6, loc="upper right",
                           facecolor="black", labelcolor="white")
            ax_hist.tick_params(colors="white", labelsize=6)
            ax_hist.set_xlabel("intensity", color="white", fontsize=7)
            ax_hist.set_ylabel("density", color="white", fontsize=7)
            for spine in ax_hist.spines.values():
                spine.set_color("gray")
            if r == 0:
                ax_hist.set_title(f"Skyra vs {label_b}", fontsize=9,
                                  color="cyan", family="monospace", pad=8)
            ax_hist.text(0.02, 0.95, method_name, transform=ax_hist.transAxes,
                         color="yellow", fontsize=7, va="top", family="monospace")

            # -- Images --
            for side, (normed, exam_id, label, color) in enumerate([
                (normed_a, exam_a, label_a, COLORS_A),
                (normed_b, exam_b, label_b, COLORS_B),
            ]):
                ax_img = axes[r, base_col + 1 + side]
                ax_img.set_facecolor("black")
                ax_img.axis("off")

                mid = normed.shape[2] // 2
                slc = normed[:, :, mid].T
                # Display normalize
                pos = slc[slc > 0]
                if len(pos) > 0:
                    p1, p99 = np.percentile(pos, [1, 99])
                    slc = np.clip(slc, p1, p99)
                    lo, hi = slc.min(), slc.max()
                    if hi > lo:
                        slc = (slc - lo) / (hi - lo)

                ax_img.imshow(slc, cmap="gray", origin="upper",
                              interpolation="bilinear", vmin=0, vmax=1)
                ax_img.text(0.02, 0.02, f"{exam_id}\n{label}",
                            ha="left", va="bottom", color=color, fontsize=7,
                            family="monospace", transform=ax_img.transAxes,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    plt.tight_layout(pad=1.0)
    # Script-local, gitignored output dir (matches view_augmentation.py's
    # convention) — training_engine/reports/ isn't gitignored, so a report
    # PNG saved there risks getting committed by accident.
    out_path = Path(__file__).parent / "output" / "normalization_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}")

    import shutil
    import subprocess
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(out_path)])
    else:
        print(f"(auto-open skipped — xdg-open not found; open manually: {out_path})")


if __name__ == "__main__":
    main()
