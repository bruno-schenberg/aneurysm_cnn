"""
Visualize tighter XY crop options on the existing 180mm NIfTIs.

Shows the mid-axial slice of each exam with a centroid-centered crop box
overlaid. Produces two separate images:
  - 160mm crop (yellow box on 180mm image)
  - 140mm crop (yellow box on 180mm image)

All NIfTIs are already at 180mm XY FOV (centroid-cropped in step 1).
The question is whether we can crop tighter without losing brain/vessels.

Usage:
    python scripts/justification/justify_tighter_xy_crop.py
    python scripts/justification/justify_tighter_xy_crop.py --n 40
    python scripts/justification/justify_tighter_xy_crop.py --class1-only
"""
import argparse
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import config
from medprep.visualize import load_mid_slice_nifti, intensity_centroid_2d
from scripts.justification.common import open_file

COLS = 5


def render(exams, crop_mm, out_path):
    n_exams = len(exams)
    n_rows = (n_exams + COLS - 1) // COLS

    panel_px = 3.0
    fig_w = COLS * panel_px
    fig_h = n_rows * panel_px

    fig, axes = plt.subplots(n_rows, COLS, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")

    if n_rows == 1:
        axes = [axes] if COLS == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    fig.suptitle(
        f"180mm NIfTI with {crop_mm:.0f}mm crop overlay (yellow=centroid crop)   "
        f"n={n_exams}",
        fontsize=7, color="white", family="monospace", y=1.002,
    )

    for i, (ax, (_, row)) in enumerate(zip(axes, exams.iterrows())):
        nifti_path = config.NIFTI_ROOT / str(int(row["class"])) / f"{row['exam_id']}.nii.gz"
        if not nifti_path.exists():
            ax.set_facecolor("black")
            ax.text(0.5, 0.5, f"{row['exam_id']}\nmissing",
                    ha="center", va="center", color="gray", fontsize=5,
                    transform=ax.transAxes)
            ax.axis("off")
            continue

        slc, sp, _shape = load_mid_slice_nifti(str(nifti_path))
        H, W = slc.shape

        ax.imshow(slc, cmap="gray", origin="upper", interpolation="bilinear")
        ax.set_facecolor("black")

        # Crop box in pixels
        box_x_px = crop_mm / sp[0]
        box_y_px = crop_mm / sp[1]

        # Center on intensity centroid
        cx, cy = intensity_centroid_2d(slc)
        left = cx - box_x_px / 2
        top = cy - box_y_px / 2

        # Clamp to image bounds
        left = max(0, min(left, W - box_x_px))
        top = max(0, min(top, H - box_y_px))

        ax.add_patch(patches.Rectangle(
            (left, top), box_x_px, box_y_px,
            linewidth=1.0, edgecolor="yellow", facecolor="none", linestyle="--",
        ))
        ax.plot(cx, cy, "+", color="yellow", markersize=3, markeredgewidth=0.6)

        cls_col = "#ff6666" if int(row["class"]) == 1 else "#aaaaaa"
        cls_label = "aneurysm" if int(row["class"]) == 1 else "control"
        ax.set_title(
            f"{row['exam_id']}  [{cls_label}]",
            fontsize=4.5, color=cls_col, family="monospace", pad=2,
        )
        ax.axis("off")

    for ax in axes[n_exams:]:
        ax.set_facecolor("black")
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}  ({n_exams} exams)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=30,
                        help="Max exams to show per image (default 30)")
    parser.add_argument("--class1-only", action="store_true",
                        help="Show only class 1 (aneurysm) exams")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    df = pd.read_csv(config.PIPELINE_REPORT)
    converted = df[df["conversion_status"] == "converted"].copy()

    if args.class1_only:
        converted = converted[converted["class"] == 1]

    # Sample if too many
    if len(converted) > args.n:
        converted = converted.sample(n=args.n, random_state=args.seed)
    converted = converted.sort_values("exam_id").reset_index(drop=True)

    out_dir = config.REPORTS_DIR / "crop_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_class1" if args.class1_only else ""

    for crop_mm in [160, 140]:
        out_path = str(out_dir / f"xy_crop_{crop_mm}mm{suffix}.png")
        render(converted, crop_mm, out_path)
        open_file(out_path)


if __name__ == "__main__":
    main()
