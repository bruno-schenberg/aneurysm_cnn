"""
Justify the cubic output shape (128³) over non-cubic alternatives.

Question: could we use 128×128×96 instead of 128³ to save memory and
compute? If the top/bottom Z slices are just air, a shorter Z makes sense.

Answer: no — at z=13 (slice 14 from the top), many exams still show
Circle of Willis anatomy. Cutting 32 slices would lose clinically
relevant vasculature. The ~26% Z padding (air) in 128³ is acceptable.

This script renders the Nth slice from the top for each exam. If that
slice contains visible anatomy, cropping Z to fewer slices would lose it.

Usage:
    python scripts/justification/justify_cube_shape.py --list
    python scripts/justification/justify_cube_shape.py --group 0
    python scripts/justification/justify_cube_shape.py --group 0 --top 14
"""
import argparse
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

import config
from scripts.justification.common import load_groups, open_file

SAMPLE_CSV = config.DATASET_CSV.parent / "sample_exams.csv"


def print_groups(groups: pd.DataFrame) -> None:
    print(f"{'idx':>4}  {'rows':>5}  {'cols':>5}  {'sx':>7}  {'FOV-xy':>10}  {'Z-med':>7}  {'N':>4}")
    for i, r in groups.iterrows():
        fov = round(r["cols_i"] * r["sx"])
        print(f"{i:>4}  {r['rows_i']:>5}  {r['cols_i']:>5}  {r['sx']:>7.4f}"
              f"  {fov:>5}x{fov:>3}mm  {r['fov_z_mm']:>6.0f}mm  {int(r['n']):>4}")


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img  = nib.load(path)
    data = img.get_fdata()
    sp   = np.array(img.header.get_zooms())   # (x, y, z) mm/voxel
    return data, sp


def norm_slice(slc: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] of non-zero voxels for display."""
    pos = slc[slc > 0]
    if len(pos) < 10:
        return slc
    p1, p99 = np.percentile(pos, [1, 99])
    return np.clip(slc, p1, p99)


COLS = 4   # panels per row; 2 panels per exam → 2 exams per row


def render(exams: pd.DataFrame, nifti_root: Path, out_path: str,
           top: int, bottom: int, side: str) -> None:
    """side: 'top' or 'bot'"""
    n_exams  = len(exams)
    n_panels = n_exams
    n_rows   = (n_panels + COLS - 1) // COLS

    panel_px = 3.5
    fig, axes = plt.subplots(n_rows, COLS, figsize=(COLS * panel_px, n_rows * panel_px))
    fig.patch.set_facecolor("black")

    if n_rows == 1:
        axes_flat = list(axes) if n_exams > 1 else [axes]
    else:
        axes_flat = [ax for row in axes for ax in row]

    if side == "top":
        title_detail = f"top-{top} (z{top - 1})"
        title_col    = "#6699ff"
    else:
        title_detail = f"bot-{bottom} (z-{bottom})"
        title_col    = "#ffcc88"

    fig.suptitle(
        f"Z-axis extremes  |  {title_detail}  |  source: {nifti_root.name}",
        fontsize=7, color=title_col, family="monospace", y=1.002,
    )

    for i, (_, exam) in enumerate(exams.iterrows()):
        ax = axes_flat[i]
        nifti_path = nifti_root / str(int(exam["class"])) / f"{exam['exam_id']}.nii.gz"

        cls_label = "aneurysm" if int(exam["class"]) == 1 else "control"
        cls_col   = "#ff6666"  if int(exam["class"]) == 1 else "#aaaaaa"

        ax.set_facecolor("black")
        ax.axis("off")

        if not nifti_path.exists():
            ax.text(0.5, 0.5, f"{exam['exam_id']}\nfile missing",
                    ha="center", va="center", color="gray",
                    fontsize=6, transform=ax.transAxes)
            continue

        data, sp = load_nifti(nifti_path)
        nz   = data.shape[2]
        z_mm = nz * float(sp[2])

        if side == "top":
            z_idx = top - 1
            slice_label = f"z{z_idx}  [top-{top}]"
        else:
            z_idx = nz - bottom
            slice_label = f"z{z_idx}  [bot-{bottom}]"

        if 0 <= z_idx < nz:
            slc = norm_slice(data[:, :, z_idx].T)
            ax.imshow(slc, cmap="gray", origin="upper", interpolation="bilinear")

        ax.set_title(
            f"{exam['exam_id']}  [{cls_label}]\n"
            f"Z={nz}vx  {z_mm:.0f}mm  sz={sp[2]:.2f}mm\n"
            f"{slice_label}",
            fontsize=5.5, color=cls_col, family="monospace", pad=2,
        )

    for ax in axes_flat[n_panels:]:
        ax.set_facecolor("black")
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}  ({n_exams} exams)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true",
                        help="List available acquisition groups and exit.")
    parser.add_argument("--group", type=int, default=None,
                        help="Only show exams from this acquisition group index (see --list).")
    parser.add_argument("--sample", action="store_true",
                        help="Restrict to exam IDs in dataset/sample_exams.csv.")
    parser.add_argument("--top", type=int, default=14,
                        help="Show the Nth slice from the top (z=N-1). Default: 14.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG (default: reports/z_extremes.png).")
    args = parser.parse_args()

    if not config.PIPELINE_REPORT.exists():
        print(f"ERROR: {config.PIPELINE_REPORT} not found. Run the pipeline first.")
        sys.exit(1)

    df                  = pd.read_csv(config.PIPELINE_REPORT)
    converted, groups, key = load_groups(df)

    if args.list or (args.group is None and not args.sample):
        print_groups(groups)
        if args.list:
            return

    if args.group is not None:
        if args.group < 0 or args.group >= len(groups):
            print(f"Group index {args.group} out of range (0–{len(groups)-1}). Use --list.")
            sys.exit(1)
        g = groups.iloc[args.group]
        mask = (
            (converted["rows_i"] == g["rows_i"]) &
            (converted["cols_i"] == g["cols_i"]) &
            (converted["sx"]     == g["sx"])
        )
        converted = converted[mask].copy()
        print(f"Group {args.group}: {len(converted)} exams  "
              f"({g['rows_i']}x{g['cols_i']}px  {g['sx']:.4f}mm/px  "
              f"Z-median={g['fov_z_mm']:.0f}mm)")

    if args.sample:
        if not SAMPLE_CSV.exists():
            print(f"ERROR: {SAMPLE_CSV} not found. "
                  "Run scripts/make_sample.py first.")
            sys.exit(1)
        sample_ids = set(pd.read_csv(SAMPLE_CSV)["exam_id"].astype(str))
        converted  = converted[converted["exam_id"].isin(sample_ids)]
        print(f"Sample mode: {len(converted)} exams")

    if converted.empty:
        print("No converted exams found.")
        sys.exit(0)

    converted  = converted.sort_values("exam_id").reset_index(drop=True)
    if args.sample:
        config.enable_test_mode()
    nifti_root = config.NIFTI_ROOT

    suffix   = f"_group{args.group}" if args.group is not None else ""
    base_out = args.out or str(config.REPORTS_DIR / f"z_top14{suffix}.png")

    render(converted, nifti_root, base_out, args.top, 0, "top")
    open_file(base_out)


if __name__ == "__main__":
    main()
