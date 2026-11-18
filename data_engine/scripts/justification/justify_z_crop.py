"""
Diagnostic script for Z-axis crop safety.

Simulates Z crops at 90mm, 80mm, and 70mm centered on the vascular
intensity centroid. Produces:

  1. CSV report with per-exam stats (centroid position, bright voxels
     lost, max intensity in discarded slabs) for each crop size.
  2. Sagittal (YZ) view at ~75% X with horizontal crop lines overlaid,
     showing what each crop keeps vs discards.
  3. Top/bottom boundary XY slices from the crop edges, so you can see
     if vessel signal is present at the cut.

--center-crop switches the sagittal view from centroid-based crop lines to
two alternative fixed-position strategies (see CENTER_STRATEGIES below):
"center" (geometric center of the volume) and "shift40" (biased toward the
bottom, closer to the Circle of Willis) — for comparing against the
centroid-based approach this script otherwise justifies.

Usage:
    python scripts/justification/justify_z_crop.py
    python scripts/justification/justify_z_crop.py --n 30 --class1-only
    python scripts/justification/justify_z_crop.py --csv-only
    python scripts/justification/justify_z_crop.py --center-crop
"""
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd

import config
from scripts.justification.common import open_file

CROP_MM = [90, 80, 70]
CROP_COLORS = {90: "#55ff55", 80: "#ffff55", 70: "#ff5555"}
CROP_LABELS = {90: "90mm", 80: "80mm", 70: "70mm"}

# Center-crop mode: position strategies
CENTER_STRATEGIES = {
    "center": 0.50,    # geometric center of volume
    "shift40": 0.40,   # biased toward bottom (CoW region)
}


# ---------------------------------------------------------------------------
# Core analysis (works on raw numpy + spacing, no plotting)
# ---------------------------------------------------------------------------

def _clamped_crop_window(center_mm: float, crop_mm: float, extent_mm: float) -> tuple[float, float]:
    """Compute a [lo_mm, hi_mm] crop window of width crop_mm centered on
    center_mm, clamped to [0, extent_mm]. When one side is clamped, the
    other side shifts to preserve the full crop_mm width where possible.
    """
    half = crop_mm / 2.0
    lo_mm = max(0, center_mm - half)
    hi_mm = min(extent_mm, center_mm + half)
    if lo_mm == 0:
        hi_mm = min(extent_mm, crop_mm)
    if hi_mm == extent_mm:
        lo_mm = max(0, extent_mm - crop_mm)
    return lo_mm, hi_mm


def z_intensity_centroid(volume: np.ndarray) -> float:
    """Return the Z-axis centroid (in voxel coords) of bright voxels.

    Uses the central 50% of XY to avoid edge artifacts, thresholds at p80
    of nonzero voxels to focus on vessels.
    """
    nx, ny, nz = volume.shape
    x0, x1 = nx // 4, nx * 3 // 4
    y0, y1 = ny // 4, ny * 3 // 4
    core = volume[x0:x1, y0:y1, :]

    # Project to Z profile (sum of bright voxels per slice)
    pos = core[core > 0]
    if len(pos) == 0:
        return nz / 2.0
    thresh = np.percentile(pos, 80)
    mask = core > thresh
    z_profile = mask.sum(axis=(0, 1)).astype(float)
    z_indices = np.arange(nz)
    total = z_profile.sum()
    if total == 0:
        return nz / 2.0
    return float(np.average(z_indices, weights=z_profile))


def analyze_one(nifti_path: str) -> dict:
    """Analyze one NIfTI for Z crop safety. Returns a stats dict."""
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    sp = img.header.get_zooms()
    nz = data.shape[2]
    sz = float(sp[2])

    z_centroid_vox = z_intensity_centroid(data)
    z_centroid_mm = z_centroid_vox * sz
    z_extent_mm = nz * sz

    # Bright voxel threshold (p95 of nonzero)
    pos = data[data > 0]
    if len(pos) == 0:
        p95_thresh = 0
        total_bright = 0
    else:
        p95_thresh = np.percentile(pos, 95)
        total_bright = int((data > p95_thresh).sum())

    result = {
        "z_spacing_mm": round(sz, 4),
        "z_slices": nz,
        "z_extent_mm": round(z_extent_mm, 1),
        "z_centroid_vox": round(z_centroid_vox, 1),
        "z_centroid_mm": round(z_centroid_mm, 1),
        "total_bright_p95": total_bright,
    }

    for crop_mm in CROP_MM:
        lo_mm, hi_mm = _clamped_crop_window(z_centroid_mm, crop_mm, z_extent_mm)

        # Convert to slice indices
        lo_sl = int(round(lo_mm / sz))
        hi_sl = int(round(hi_mm / sz))
        lo_sl = max(0, min(lo_sl, nz))
        hi_sl = max(0, min(hi_sl, nz))

        # Bright voxels in discarded top/bottom slabs
        bottom_slab = data[:, :, :lo_sl] if lo_sl > 0 else np.array([])
        top_slab = data[:, :, hi_sl:] if hi_sl < nz else np.array([])

        bright_bottom = int((bottom_slab > p95_thresh).sum()) if bottom_slab.size > 0 else 0
        bright_top = int((top_slab > p95_thresh).sum()) if top_slab.size > 0 else 0
        bright_lost = bright_bottom + bright_top
        bright_lost_pct = (bright_lost / total_bright * 100) if total_bright > 0 else 0.0

        max_bottom = float(bottom_slab.max()) if bottom_slab.size > 0 else 0.0
        max_top = float(top_slab.max()) if top_slab.size > 0 else 0.0
        max_discarded = max(max_bottom, max_top)

        # Vessel-like voxel fraction in discarded slabs (>p90 of whole volume)
        p90_thresh = np.percentile(pos, 90) if len(pos) > 0 else 0
        discard_voxels = bottom_slab.size + top_slab.size
        vessel_bottom = int((bottom_slab > p90_thresh).sum()) if bottom_slab.size > 0 else 0
        vessel_top = int((top_slab > p90_thresh).sum()) if top_slab.size > 0 else 0
        vessel_in_discard_pct = ((vessel_bottom + vessel_top) / discard_voxels * 100) if discard_voxels > 0 else 0.0

        c = crop_mm
        result[f"crop{c}_lo_mm"] = round(lo_mm, 1)
        result[f"crop{c}_hi_mm"] = round(hi_mm, 1)
        result[f"crop{c}_lo_sl"] = lo_sl
        result[f"crop{c}_hi_sl"] = hi_sl
        result[f"crop{c}_slices_kept"] = hi_sl - lo_sl
        result[f"crop{c}_bright_lost_pct"] = round(bright_lost_pct, 2)
        result[f"crop{c}_max_discarded"] = round(max_discarded, 2)
        result[f"crop{c}_vessel_discard_pct"] = round(vessel_in_discard_pct, 4)
        result[f"crop{c}_flag"] = bright_lost_pct > 2.0

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def load_sagittal_slice(path, x_frac=0.75):
    """Load a sagittal (YZ) slice at the given X fraction."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    sp = img.header.get_zooms()
    xi = int(x_frac * data.shape[0])
    xi = max(0, min(xi, data.shape[0] - 1))
    slc = data[xi, :, :]  # (Y, Z)
    # Contrast stretch
    pos = slc[slc > 0]
    if len(pos) > 0:
        p1, p99 = np.percentile(pos, [1, 99])
        slc = np.clip(slc, p1, p99)
    return slc, sp, data


def render_sagittal(exams, out_path, cols=5):
    """Render sagittal views with Z crop lines for each exam."""
    n_exams = len(exams)
    n_rows = (n_exams + cols - 1) // cols

    panel_w, panel_h = 3.0, 2.0  # sagittal panels are wider than tall (Y > Z typically)
    fig_w = cols * panel_w
    fig_h = n_rows * panel_h

    fig, axes = plt.subplots(n_rows, cols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")

    if n_rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    legend_parts = "  ".join(
        f"{CROP_LABELS[c]}={CROP_COLORS[c]}" for c in CROP_MM
    )
    fig.suptitle(
        f"Sagittal view (YZ at 75% X) with Z crop boundaries   "
        f"{legend_parts}   n={n_exams}",
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

        slc, sp, data = load_sagittal_slice(str(nifti_path))
        ny, nz = slc.shape
        sz = float(sp[2])

        # Display: Y on x-axis, Z on y-axis (origin=lower so Z=0 is bottom)
        ax.imshow(slc.T, cmap="gray", origin="lower", interpolation="bilinear",
                  aspect="auto")
        ax.set_facecolor("black")

        # Compute centroid
        z_centroid_vox = z_intensity_centroid(data)
        z_centroid_mm = z_centroid_vox * sz
        z_extent_mm = nz * sz

        # Draw crop lines
        for crop_mm in CROP_MM:
            lo_mm, hi_mm = _clamped_crop_window(z_centroid_mm, crop_mm, z_extent_mm)

            lo_sl = lo_mm / sz
            hi_sl = hi_mm / sz

            color = CROP_COLORS[crop_mm]
            ax.axhline(lo_sl, color=color, linewidth=0.8, linestyle="--", alpha=0.9)
            ax.axhline(hi_sl, color=color, linewidth=0.8, linestyle="--", alpha=0.9)

        # Centroid marker
        ax.axhline(z_centroid_vox, color="cyan", linewidth=0.5, linestyle=":",
                    alpha=0.7)

        cls_col = "#ff6666" if int(row["class"]) == 1 else "#aaaaaa"
        cls_label = "A" if int(row["class"]) == 1 else "C"
        ax.set_title(
            f"{row['exam_id']} [{cls_label}] z={z_extent_mm:.0f}mm",
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


def render_sagittal_center_crop(exams, out_path, cols=5):
    """Render sagittal views with CENTER-CROP Z lines (not centroid-based).

    Shows two strategies:
      - center (50%): green lines — geometric center of volume
      - shift40 (40%): orange lines — biased toward CoW
    """
    n_exams = len(exams)
    n_rows = (n_exams + cols - 1) // cols

    panel_w, panel_h = 3.0, 2.0
    fig_w = cols * panel_w
    fig_h = n_rows * panel_h

    fig, axes = plt.subplots(n_rows, cols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")

    if n_rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    fig.suptitle(
        "Sagittal (YZ 75% X) — CENTER CROP lines   "
        "green=center(50%)  orange=shift40(40%)   "
        "solid=90mm  dashed=80mm  dotted=70mm",
        fontsize=6, color="white", family="monospace", y=1.002,
    )

    crop_styles = {90: "-", 80: "--", 70: ":"}
    strategy_colors = {"center": "#55ff55", "shift40": "#ff9933"}

    for i, (ax, (_, row)) in enumerate(zip(axes, exams.iterrows())):
        nifti_path = config.NIFTI_ROOT / str(int(row["class"])) / f"{row['exam_id']}.nii.gz"
        if not nifti_path.exists():
            ax.set_facecolor("black")
            ax.text(0.5, 0.5, f"{row['exam_id']}\nmissing",
                    ha="center", va="center", color="gray", fontsize=5,
                    transform=ax.transAxes)
            ax.axis("off")
            continue

        slc, sp, data = load_sagittal_slice(str(nifti_path))
        ny, nz = slc.shape
        sz = float(sp[2])
        z_extent_mm = nz * sz

        ax.imshow(slc.T, cmap="gray", origin="lower", interpolation="bilinear",
                  aspect="auto")
        ax.set_facecolor("black")

        for crop_mm in CROP_MM:
            for strategy_name, center_frac in CENTER_STRATEGIES.items():
                center_mm = z_extent_mm * center_frac
                lo_mm, hi_mm = _clamped_crop_window(center_mm, crop_mm, z_extent_mm)

                lo_sl = lo_mm / sz
                hi_sl = hi_mm / sz

                color = strategy_colors[strategy_name]
                style = crop_styles[crop_mm]
                ax.axhline(lo_sl, color=color, linewidth=0.8, linestyle=style, alpha=0.9)
                ax.axhline(hi_sl, color=color, linewidth=0.8, linestyle=style, alpha=0.9)

        cls_col = "#ff6666" if int(row["class"]) == 1 else "#aaaaaa"
        cls_label = "A" if int(row["class"]) == 1 else "C"
        ax.set_title(
            f"{row['exam_id']} [{cls_label}] z={z_extent_mm:.0f}mm",
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


def render_boundary_slices(exams, crop_mm, out_path, cols=5):
    """For each exam, show the top and bottom XY slices at the crop boundary."""
    n_exams = len(exams)
    n_rows = (n_exams + cols - 1) // cols

    panel_px = 2.5
    # Two sub-columns per exam: bottom slice | top slice
    fig_w = cols * panel_px * 2
    fig_h = n_rows * panel_px

    fig, axes = plt.subplots(n_rows, cols * 2, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")

    if n_rows == 1 and cols * 2 == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    fig.suptitle(
        f"Boundary XY slices at {crop_mm}mm Z crop edges   "
        f"left=bottom boundary  right=top boundary   n={n_exams}",
        fontsize=7, color="white", family="monospace", y=1.002,
    )

    for i, (_, row) in enumerate(exams.iterrows()):
        ax_bot = axes[i * 2]
        ax_top = axes[i * 2 + 1]

        nifti_path = config.NIFTI_ROOT / str(int(row["class"])) / f"{row['exam_id']}.nii.gz"
        if not nifti_path.exists():
            for ax in (ax_bot, ax_top):
                ax.set_facecolor("black")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="gray", fontsize=5, transform=ax.transAxes)
                ax.axis("off")
            continue

        img = nib.load(str(nifti_path))
        data = img.get_fdata(dtype=np.float32)
        sp = img.header.get_zooms()
        nz = data.shape[2]
        sz = float(sp[2])

        z_centroid_vox = z_intensity_centroid(data)
        z_centroid_mm = z_centroid_vox * sz
        z_extent_mm = nz * sz

        lo_mm, hi_mm = _clamped_crop_window(z_centroid_mm, crop_mm, z_extent_mm)

        lo_sl = max(0, min(int(round(lo_mm / sz)), nz - 1))
        hi_sl = max(0, min(int(round(hi_mm / sz)) - 1, nz - 1))

        # Global contrast reference
        pos = data[data > 0]
        if len(pos) > 0:
            p1, p99 = np.percentile(pos, [1, 99])
        else:
            p1, p99 = 0, 1

        for ax, sl_idx, label in [(ax_bot, lo_sl, "bottom"), (ax_top, hi_sl, "top")]:
            slc = data[:, :, sl_idx].T  # (Y, X)
            slc = np.clip(slc, p1, p99)
            ax.imshow(slc, cmap="gray", origin="upper", interpolation="bilinear")
            ax.set_facecolor("black")

            # Check for bright signal in this slice
            raw_slc = data[:, :, sl_idx]
            if len(pos) > 0:
                p90 = np.percentile(pos, 90)
                bright_pct = (raw_slc > p90).sum() / raw_slc.size * 100
            else:
                bright_pct = 0

            # Color title by brightness: red if >1% bright voxels
            title_col = "#ff5555" if bright_pct > 1.0 else "#aaaaaa"
            cls_label = "A" if int(row["class"]) == 1 else "C"
            ax.set_title(
                f"{row['exam_id']} [{cls_label}]\n{label} z={sl_idx} "
                f"bright={bright_pct:.1f}%",
                fontsize=3.5, color=title_col, family="monospace", pad=2,
            )
            ax.axis("off")

    # Hide unused
    for ax in axes[n_exams * 2:]:
        ax.set_facecolor("black")
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved {out_path}  ({n_exams} exams)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=30,
                        help="Max exams for visual reports (default 30)")
    parser.add_argument("--class1-only", action="store_true",
                        help="Visuals for class 1 (aneurysm) exams only")
    parser.add_argument("--csv-only", action="store_true",
                        help="Only produce the CSV report (all exams), skip visuals")
    parser.add_argument("--visuals-only", action="store_true",
                        help="Only produce visuals, skip CSV")
    parser.add_argument("--center-crop", action="store_true",
                        help="Show center-crop lines instead of centroid-based")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(config.PIPELINE_REPORT)
    converted = df[df["conversion_status"] == "converted"].copy()

    out_dir = config.REPORTS_DIR / "z_crop"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV report (all exams) ---
    if not args.visuals_only:
        print(f"Analyzing {len(converted)} exams for Z crop safety...")
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for _, row in converted.iterrows():
            nifti_path = str(
                config.NIFTI_ROOT / str(int(row["class"])) / f"{row['exam_id']}.nii.gz"
            )
            tasks.append((row["exam_id"], int(row["class"]), nifti_path))

        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(analyze_one, t[2]): t for t in tasks
            }
            done = 0
            for future in as_completed(futures):
                exam_id, cls, path = futures[future]
                done += 1
                try:
                    stats = future.result()
                    stats["exam_id"] = exam_id
                    stats["class"] = cls
                    rows.append(stats)
                except Exception as e:
                    print(f"  FAILED {exam_id}: {e}")
                    rows.append({"exam_id": exam_id, "class": cls, "error": str(e)})
                if done % 100 == 0:
                    print(f"  {done}/{len(tasks)} done")

        report = pd.DataFrame(rows)
        # Move exam_id and class to front
        cols = ["exam_id", "class"] + [c for c in report.columns if c not in ("exam_id", "class")]
        report = report[cols].sort_values("exam_id").reset_index(drop=True)

        csv_path = out_dir / "z_crop_diagnostic.csv"
        report.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}  ({len(report)} exams)")

        # Summary
        print("\n=== Z crop safety summary ===")
        for crop_mm in CROP_MM:
            c = crop_mm
            flagged = report[report.get(f"crop{c}_flag", False) == True]
            mean_lost = report[f"crop{c}_bright_lost_pct"].mean()
            max_lost = report[f"crop{c}_bright_lost_pct"].max()
            mean_vessel = report[f"crop{c}_vessel_discard_pct"].mean()
            print(f"  {crop_mm}mm: {len(flagged)} flagged (>2% bright lost)  "
                  f"mean_bright_lost={mean_lost:.2f}%  max={max_lost:.2f}%  "
                  f"mean_vessel_in_discard={mean_vessel:.4f}%")
            if len(flagged) > 0:
                print(f"    Flagged exams: {', '.join(flagged['exam_id'].tolist())}")

    if args.csv_only:
        return

    # --- Visual reports ---
    vis_df = converted.copy()
    if args.class1_only:
        vis_df = vis_df[vis_df["class"] == 1]
    if len(vis_df) > args.n:
        vis_df = vis_df.sample(n=args.n, random_state=args.seed)
    vis_df = vis_df.sort_values("exam_id").reset_index(drop=True)

    suffix = "_class1" if args.class1_only else ""

    if args.center_crop:
        # Center-crop mode: show positional crop lines
        sag_path = str(out_dir / f"sagittal_center_crop{suffix}.png")
        render_sagittal_center_crop(vis_df, sag_path)
        open_file(sag_path)
    else:
        # Original centroid-based mode
        sag_path = str(out_dir / f"sagittal_z_crops{suffix}.png")
        render_sagittal(vis_df, sag_path)
        open_file(sag_path)

        # Boundary slices for each crop size
        for crop_mm in CROP_MM:
            bnd_path = str(out_dir / f"boundary_slices_{crop_mm}mm{suffix}.png")
            render_boundary_slices(vis_df, crop_mm, bnd_path)
            open_file(bnd_path)


if __name__ == "__main__":
    main()
