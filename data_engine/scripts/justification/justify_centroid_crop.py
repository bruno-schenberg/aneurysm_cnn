"""
Justify the centroid crop strategy over geometric-centre crop.

Problem: MRA exams have FOVs from 180–240mm. To build a uniform dataset
we crop to 180mm, but naive geometric cropping cuts brain tissue when the
head is off-centre in the FOV.

Solution: crop around the intensity centroid (centre of mass of bright
voxels). This tracks the brain position regardless of head placement,
keeping the Circle of Willis centred in the output volume.

Each panel overlays two 180mm crop boxes on the middle axial slice:
  red    — geometric centre (assumes head is centred in FOV)
  yellow — intensity centroid (chosen: tracks the brain mass)
  green  — FOV already ≤ 180mm, no crop needed

Usage:
    python scripts/justification/justify_centroid_crop.py --list
    python scripts/justification/justify_centroid_crop.py --group 2
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
from scripts.justification.common import load_groups, open_file

COLS = 4


def print_groups(groups):
    print(f"{'idx':>4}  {'rows':>5}  {'cols':>5}  {'sx':>7}  {'FOV-xy':>10}  {'N':>4}")
    for i, r in groups.iterrows():
        fov = round(r["cols_i"] * r["sx"])
        print(f"{i:>4}  {r['rows_i']:>5}  {r['cols_i']:>5}  {r['sx']:>7.4f}  {fov:>5}x{fov:>3}mm  {int(r['n']):>4}")


def render_group(exams, group_row, crop_mm, out_path):
    n_exams = len(exams)
    n_rows  = (n_exams + COLS - 1) // COLS

    panel_px = 3.5          # inches per panel
    fig_w    = COLS * panel_px
    fig_h    = n_rows * panel_px

    fig, axes = plt.subplots(n_rows, COLS, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")

    # Flatten axes even for single row
    if n_rows == 1:
        axes = [axes] if COLS == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    fov_xy = round(group_row["cols_i"] * group_row["sx"])
    header = (
        f"Group  {group_row['rows_i']}x{group_row['cols_i']}px  "
        f"{group_row['sx']:.4f}mm/px  FOV {fov_xy}x{fov_xy}mm  "
        f"crop={crop_mm:.0f}mm   red=geometric centre   yellow=intensity centroid   green=no crop needed   "
        f"n={int(group_row['n'])} exams"
    )
    fig.suptitle(header, fontsize=7, color="white", family="monospace", y=1.002)

    for i, (ax, (_, row)) in enumerate(zip(axes, exams.iterrows())):
        nifti_path = config.NIFTI_ROOT / str(int(row["class"])) / f"{row['exam_id']}.nii.gz"
        if not nifti_path.exists():
            ax.set_facecolor("black")
            ax.text(0.5, 0.5, f"{row['exam_id']}\nfile missing",
                    ha="center", va="center", color="gray", fontsize=6, transform=ax.transAxes)
            ax.axis("off")
            continue

        slc, sp, _shape = load_mid_slice_nifti(str(nifti_path))
        H, W    = slc.shape

        # Embed the slice in a padded canvas so the image border is always visible
        PAD  = max(4, int(min(H, W) * 0.04))
        canvas = np.zeros((H + 2 * PAD, W + 2 * PAD), dtype=slc.dtype)
        canvas[PAD:PAD + H, PAD:PAD + W] = slc
        ax.imshow(canvas, cmap="gray", origin="upper", interpolation="bilinear")
        ax.set_facecolor("black")

        # White border — original image boundary
        ax.add_patch(patches.Rectangle(
            (PAD, PAD), W, H,
            linewidth=1.0, edgecolor="white", facecolor="none", linestyle="-",
        ))

        fov_mm   = W * sp[0]           # physical XY extent of this exam
        needs_crop = fov_mm > crop_mm  # only shift when there is excess FOV to trim

        box_x_px = min(crop_mm / sp[0], W)
        box_y_px = min(crop_mm / sp[1], H)

        if needs_crop:
            # Red box — geometric centre (offset by PAD)
            left_geo = PAD + (W - box_x_px) / 2
            top_geo  = PAD + (H - box_y_px) / 2
            ax.add_patch(patches.Rectangle(
                (left_geo, top_geo), box_x_px, box_y_px,
                linewidth=1.2, edgecolor="red", facecolor="none", linestyle="--",
            ))

            # Yellow box — intensity centroid (offset by PAD)
            cx, cy   = intensity_centroid_2d(slc)
            left_cen = PAD + cx - box_x_px / 2
            top_cen  = PAD + cy - box_y_px / 2
            ax.add_patch(patches.Rectangle(
                (left_cen, top_cen), box_x_px, box_y_px,
                linewidth=1.2, edgecolor="yellow", facecolor="none", linestyle="--",
            ))
            ax.plot(PAD + cx, PAD + cy, "+", color="yellow", markersize=4, markeredgewidth=0.8)
        else:
            # FOV already at or below target — no crop needed, just mark it
            ax.add_patch(patches.Rectangle(
                (PAD, PAD), W, H,
                linewidth=1.2, edgecolor="green", facecolor="none", linestyle="--",
            ))

        # Title: exam ID + class, colour-coded
        cls_col  = "#ff6666" if int(row["class"]) == 1 else "#aaaaaa"
        cls_label = "aneurysm" if int(row["class"]) == 1 else "control"
        ax.set_title(
            f"{row['exam_id']}  [{cls_label}]",
            fontsize=5.5, color=cls_col, family="monospace", pad=2,
        )
        ax.axis("off")

    # Hide unused panels
    for ax in axes[n_exams:]:
        ax.set_facecolor("black")
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}  ({n_exams} exams, {n_rows} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list",  action="store_true", help="List available groups and exit")
    parser.add_argument("--group", type=int, default=None, help="Group index to render (see --list)")
    parser.add_argument("--crop",  type=float, default=180.0, help="Crop box size in mm (default 180)")
    parser.add_argument("--out",   type=str, default=None, help="Output PNG path (default /tmp/group_slices_N.png)")
    args = parser.parse_args()

    df = pd.read_csv(config.PIPELINE_REPORT)
    converted, groups, key = load_groups(df)

    if args.list or args.group is None:
        print_groups(groups)
        return

    if args.group < 0 or args.group >= len(groups):
        print(f"Group index {args.group} out of range (0–{len(groups)-1}). Use --list to see options.")
        sys.exit(1)

    group_row = groups.iloc[args.group]
    mask = (
        (converted["rows_i"] == group_row["rows_i"]) &
        (converted["cols_i"] == group_row["cols_i"]) &
        (converted["sx"]     == group_row["sx"])
    )
    exams = converted[mask].sort_values("exam_id").reset_index(drop=True)

    out_dir  = config.REPORTS_DIR / "group_slices"
    out_dir.mkdir(exist_ok=True)
    fov_xy   = round(group_row["cols_i"] * group_row["sx"])
    out_path = args.out or str(out_dir / f"group_{args.group:02d}_{group_row['rows_i']}x{group_row['cols_i']}_{fov_xy}mm.png")
    render_group(exams, group_row, args.crop, out_path)
    open_file(out_path)


if __name__ == "__main__":
    main()
