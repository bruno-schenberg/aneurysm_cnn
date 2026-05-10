"""
analyse_nifti_survey.py

Deep analysis of nifti_survey.csv to inform resize strategy decisions.

Outputs:
  - Console summary stats per class
  - Simulation of padding/cropping waste for candidate target shapes
  - Plots saved to data_engine/outputs/survey_plots/

Usage:
    python data_engine/diagnostics/analyse_nifti_survey.py
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "outputs/nifti_survey.csv")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "outputs/survey_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "exam": r["exam"],
                "class": r["class"],
                "dim_x": int(r["dim_x"]),
                "dim_y": int(r["dim_y"]),
                "dim_z": int(r["dim_z"]),
                "sx": float(r["spacing_x_mm"]),
                "sy": float(r["spacing_y_mm"]),
                "sz": float(r["spacing_z_mm"]),
                "fov_x": float(r["size_x_mm"]),
                "fov_y": float(r["size_y_mm"]),
                "fov_z": float(r["size_z_mm"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def col(rows, key):
    return np.array([r[key] for r in rows])


def percentiles(arr, ps=(5, 25, 50, 75, 95)):
    return {p: float(np.percentile(arr, p)) for p in ps}


def class_split(rows):
    return (
        [r for r in rows if r["class"] == "0"],
        [r for r in rows if r["class"] == "1"],
    )


def print_stat_block(label, arr):
    p = percentiles(arr)
    print(f"    {label:18s}  min={arr.min():.3f}  p5={p[5]:.3f}  p25={p[25]:.3f}  "
          f"median={p[50]:.3f}  p75={p[75]:.3f}  p95={p[95]:.3f}  max={arr.max():.3f}")


# ---------------------------------------------------------------------------
# 1. Per-class summary
# ---------------------------------------------------------------------------

def per_class_summary(rows):
    neg, pos = class_split(rows)
    print("=" * 80)
    print(f"DATASET OVERVIEW   total={len(rows)}  class0={len(neg)}  class1={len(pos)}")
    print("=" * 80)

    for label, subset in [("Class 0 (healthy)", neg), ("Class 1 (aneurysm)", pos)]:
        print(f"\n{label}  (n={len(subset)})")
        for key, name in [
            ("dim_x", "dim_x (vox)"), ("dim_y", "dim_y (vox)"), ("dim_z", "dim_z (vox)"),
            ("sx", "spacing_x (mm)"), ("sy", "spacing_y (mm)"), ("sz", "spacing_z (mm)"),
            ("fov_x", "FOV_x (mm)"), ("fov_y", "FOV_y (mm)"), ("fov_z", "FOV_z (mm)"),
        ]:
            print_stat_block(name, col(subset, key))


# ---------------------------------------------------------------------------
# 2. Spacing anisotropy
# ---------------------------------------------------------------------------

def spacing_anisotropy(rows):
    print("\n" + "=" * 80)
    print("SPACING ANISOTROPY")
    print("=" * 80)

    sx = col(rows, "sx")
    sz = col(rows, "sz")
    ratio = sz / sx  # >1 means Z is coarser than in-plane

    bins = [(0, 1.1), (1.1, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]
    print("\n  Z/XY spacing ratio distribution (ratio >1 → Z coarser than in-plane):")
    for lo, hi in bins:
        n = int(np.sum((ratio >= lo) & (ratio < hi)))
        pct = 100 * n / len(rows)
        bar = "#" * int(pct / 2)
        print(f"    {lo:.1f}–{hi:.1f}x  {n:4d} ({pct:4.1f}%)  {bar}")

    print(f"\n  Median Z/XY ratio: {np.median(ratio):.2f}x")
    print(f"  p95 Z/XY ratio:    {np.percentile(ratio, 95):.2f}x")


# ---------------------------------------------------------------------------
# 3. Candidate resize strategy simulation
# ---------------------------------------------------------------------------

# Each strategy is (name, target_shape, resample_to_iso_mm, post_op)
# resample_to_iso_mm=None  → use native spacing (no resampling)
# resample_to_iso_mm=X     → resample to X mm isotropic first
# resample_to_iso_mm="per" → resample each exam to its own min spacing (per-exam isotropic)
# post_op: "crop" or "shrink"
STRATEGIES = [
    # --- Current 8 datasets ---
    ("128³  iso-1mm  crop",          (128, 128, 128), 1.0,   "crop"),
    ("128³  iso-1mm  shrink",        (128, 128, 128), 1.0,   "shrink"),
    ("128³  native   crop",          (128, 128, 128), None,  "crop"),
    ("128³  native   shrink",        (128, 128, 128), None,  "shrink"),
    ("256x256x128  iso-1mm  crop",   (256, 256, 128), 1.0,   "crop"),
    ("256x256x128  iso-1mm  shrink", (256, 256, 128), 1.0,   "shrink"),
    ("256x256x128  native   crop",   (256, 256, 128), None,  "crop"),
    ("256x256x128  native   shrink", (256, 256, 128), None,  "shrink"),
    # --- Candidate new shapes ---
    ("256³  iso-1mm  crop",          (256, 256, 256), 1.0,   "crop"),
    ("256³  iso-1mm  shrink",        (256, 256, 256), 1.0,   "shrink"),
    ("256x256x192  iso-1mm  crop",   (256, 256, 192), 1.0,   "crop"),
    ("256x256x192  iso-1mm  shrink", (256, 256, 192), 1.0,   "shrink"),
    # --- Per-exam isotropic (each exam resampled to its own min spacing) ---
    ("256³  per-exam-iso  crop",     (256, 256, 256), "per", "crop"),
    ("256³  per-exam-iso  shrink",   (256, 256, 256), "per", "shrink"),
    ("256x256x192  per-exam-iso  crop",  (256, 256, 192), "per", "crop"),
    ("256x256x192  per-exam-iso  shrink",(256, 256, 192), "per", "shrink"),
]


def simulate_one(r, target_shape, resample_to_iso_mm, post_op):
    """
    Simulate one exam through: optional resample → crop or shrink → pad to target.

    crop:   center-crop to target; voxels outside are discarded.
    shrink: scale volume so largest dim fits in target, then pad remainder with zeros.
            No information is discarded; effective voxel size increases.
    """
    tx, ty, tz = target_shape

    if resample_to_iso_mm == "per":
        iso_mm = min(r["sx"], r["sy"], r["sz"])
    elif resample_to_iso_mm is not None:
        iso_mm = resample_to_iso_mm
    else:
        iso_mm = None

    if iso_mm is not None:
        nx = r["fov_x"] / iso_mm
        ny = r["fov_y"] / iso_mm
        nz = r["fov_z"] / iso_mm
        effective_spacing = iso_mm
    else:
        nx, ny, nz = r["dim_x"], r["dim_y"], r["dim_z"]
        effective_spacing = min(r["sx"], r["sy"], r["sz"])

    target_vol = tx * ty * tz

    if post_op == "crop":
        kept_x = min(nx, tx)
        kept_y = min(ny, ty)
        kept_z = min(nz, tz)
        original_vol = nx * ny * nz
        kept_vol = kept_x * kept_y * kept_z
        cropped_frac = max(0.0, (original_vol - kept_vol) / original_vol)
        padding_frac = max(0.0, (target_vol - kept_vol) / target_vol)
        fits = (nx <= tx) and (ny <= ty) and (nz <= tz)
        eff_sp_final = effective_spacing

    else:  # shrink
        # Scale so the largest dim fits inside the target on each axis independently
        # (i.e., scale per-axis so each dim fits, no cropping ever happens)
        scale = min(tx / nx, ty / ny, tz / nz)
        # Only shrink, never upscale
        scale = min(scale, 1.0)
        sx_f = nx * scale
        sy_f = ny * scale
        sz_f = nz * scale
        kept_vol = sx_f * sy_f * sz_f
        cropped_frac = 0.0  # shrink never discards voxels
        padding_frac = max(0.0, (target_vol - kept_vol) / target_vol)
        fits = True  # shrink always fits
        eff_sp_final = effective_spacing / scale if scale > 0 else effective_spacing

    return {
        "exam": r["exam"],
        "class": r["class"],
        "fits": fits,
        "cropped_frac": cropped_frac,
        "padding_frac": padding_frac,
        "effective_spacing_mm": eff_sp_final,
    }


def simulate_strategy(rows, target_shape, resample_to_iso_mm, post_op):
    return [simulate_one(r, target_shape, resample_to_iso_mm, post_op) for r in rows]


def strategy_summary(name, sim_results):
    fits = sum(1 for r in sim_results if r["fits"])
    needs_crop = len(sim_results) - fits
    pad_fracs = np.array([r["padding_frac"] for r in sim_results])
    crop_fracs = np.array([r["cropped_frac"] for r in sim_results])

    heavy_pad = int(np.sum(pad_fracs > 0.30))   # >30% padding
    heavy_crop = int(np.sum(crop_fracs > 0.10))  # >10% cropped away

    return {
        "name": name,
        "fits_n": fits,
        "needs_crop_n": needs_crop,
        "fits_pct": 100 * fits / len(sim_results),
        "median_pad_pct": float(np.median(pad_fracs) * 100),
        "p95_pad_pct": float(np.percentile(pad_fracs, 95) * 100),
        "median_crop_pct": float(np.median(crop_fracs) * 100),
        "p95_crop_pct": float(np.percentile(crop_fracs, 95) * 100),
        "heavy_pad_n": heavy_pad,
        "heavy_crop_n": heavy_crop,
    }


def simulate_all(rows):
    print("\n" + "=" * 80)
    print("RESIZE STRATEGY SIMULATION")
    print("=" * 80)
    print(f"\n{'Strategy':<32} {'Fits':>5} {'NeedCrop':>9} "
          f"{'MedPad%':>8} {'p95Pad%':>8} "
          f"{'MedCrop%':>9} {'p95Crop%':>9} "
          f"{'HeavyPad':>9} {'HeavyCrop':>10}")
    print("-" * 110)

    all_sims = {}
    for name, shape, iso in STRATEGIES:
        sim = simulate_strategy(rows, shape, iso)
        s = strategy_summary(name, sim)
        all_sims[name] = sim
        print(
            f"  {s['name']:<30} {s['fits_n']:>5} ({s['fits_pct']:4.0f}%)  "
            f"{s['needs_crop_n']:>6}   "
            f"{s['median_pad_pct']:>7.1f}%  {s['p95_pad_pct']:>7.1f}%  "
            f"{s['median_crop_pct']:>8.1f}%  {s['p95_crop_pct']:>8.1f}%  "
            f"{s['heavy_pad_n']:>8}   {s['heavy_crop_n']:>8}"
        )

    return all_sims


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def plots(rows):
    neg, pos = class_split(rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("NIfTI Survey — Raw Exam Properties (n=745)", fontsize=13)

    plot_specs = [
        ("fov_x", "FOV X (mm)", 0, 0),
        ("fov_y", "FOV Y (mm)", 0, 1),
        ("fov_z", "FOV Z (mm)", 0, 2),
        ("sx",    "Spacing XY (mm)", 1, 0),
        ("sz",    "Spacing Z (mm)",  1, 1),
    ]

    for key, title, ri, ci in plot_specs:
        ax = axes[ri][ci]
        v0 = col(neg, key)
        v1 = col(pos, key)
        bins = np.linspace(min(v0.min(), v1.min()), max(v0.max(), v1.max()), 40)
        ax.hist(v0, bins=bins, alpha=0.6, label="Class 0", color="steelblue")
        ax.hist(v1, bins=bins, alpha=0.6, label="Class 1", color="tomato")
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # Z/XY ratio
    ax = axes[1][2]
    ratio0 = col(neg, "sz") / col(neg, "sx")
    ratio1 = col(pos, "sz") / col(pos, "sx")
    bins = np.linspace(0, max(ratio0.max(), ratio1.max()) * 1.05, 40)
    ax.hist(ratio0, bins=bins, alpha=0.6, label="Class 0", color="steelblue")
    ax.hist(ratio1, bins=bins, alpha=0.6, label="Class 1", color="tomato")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="isotropic")
    ax.set_title("Z/XY Spacing Ratio")
    ax.set_xlabel("sz / sx  (>1 = coarser Z)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "01_raw_properties.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nSaved: {out}")

    # --- Padding vs cropping scatter for key strategies ---
    key_strategies = [
        ("128³  iso-1mm  crop",   "128³ iso-1mm"),
        ("256x256x128  iso-1mm  crop",  "256x256x128 iso-1mm"),
        ("256³  iso-1mm  crop",   "256³ iso-1mm"),
        ("256x256x192  iso-1mm",  "256x256x192 iso-1mm"),
        ("256x256x192  iso-0.5mm","256x256x192 iso-0.5mm"),
        ("320x320x160  iso-0.5mm","320x320x160 iso-0.5mm"),
    ]

    all_sims_local = {}
    for internal_name, _ in key_strategies:
        for name, shape, iso in STRATEGIES:
            if name == internal_name:
                all_sims_local[internal_name] = simulate_strategy(rows, shape, iso)
                break

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Padding vs Crop waste per strategy", fontsize=13)

    for ax, (internal_name, display_name) in zip(axes.flat, key_strategies):
        sim = all_sims_local[internal_name]
        neg_sim = [s for s in sim if s["class"] == "0"]
        pos_sim = [s for s in sim if s["class"] == "1"]

        for subset, color, label in [(neg_sim, "steelblue", "Class 0"),
                                     (pos_sim, "tomato", "Class 1")]:
            pad = [s["padding_frac"] * 100 for s in subset]
            crop = [s["cropped_frac"] * 100 for s in subset]
            ax.scatter(pad, crop, alpha=0.3, s=6, color=color, label=label)

        ax.axvline(30, color="orange", linestyle="--", linewidth=0.8, label="30% pad")
        ax.axhline(10, color="red", linestyle="--", linewidth=0.8, label="10% crop")
        ax.set_title(display_name, fontsize=9)
        ax.set_xlabel("Padding %")
        ax.set_ylabel("Cropped %")
        ax.set_xlim(-1, 80)
        ax.set_ylim(-0.5, 60)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "02_padding_vs_crop.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")

    # --- FOV coverage: what % of exams fit in each candidate XY FOV ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Physical FOV coverage at 0.5mm and 1mm isotropic resampling")

    for ax, iso_mm, title in [(ax1, 0.5, "0.5 mm isotropic"), (ax2, 1.0, "1.0 mm isotropic")]:
        targets = [64, 96, 128, 160, 192, 224, 256, 320, 384]
        for axis, key, color, label in [
            ("X", "fov_x", "steelblue", "X-axis"),
            ("Y", "fov_y", "seagreen",  "Y-axis"),
            ("Z", "fov_z", "tomato",    "Z-axis"),
        ]:
            coverage = []
            for t in targets:
                fov_covered_mm = t * iso_mm
                fits = np.mean(col(rows, key) <= fov_covered_mm) * 100
                coverage.append(fits)
            ax.plot(targets, coverage, marker="o", color=color, label=label)

        ax.axhline(90, color="gray", linestyle="--", linewidth=0.7, label="90% coverage")
        ax.axhline(95, color="black", linestyle="--", linewidth=0.7, label="95% coverage")
        ax.set_title(title)
        ax.set_xlabel("Target voxels per axis")
        ax.set_ylabel("% exams that fit without cropping")
        ax.set_xticks(targets)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "03_fov_coverage.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Coverage table — what shape covers what % of exams without cropping
# ---------------------------------------------------------------------------

def coverage_table(rows):
    print("\n" + "=" * 80)
    print("FOV COVERAGE TABLE  (% of exams that fit WITHOUT cropping per axis)")
    print("  → 100% means no exam would be cropped on that axis")
    print("=" * 80)

    isos = [0.5, 1.0]
    targets = [128, 160, 192, 224, 256, 320]

    for iso in isos:
        print(f"\n  Isotropic {iso} mm resampling:")
        print(f"  {'Voxels':>7}  {'FOV(mm)':>8}  {'X%':>5}  {'Y%':>5}  {'Z%':>5}  {'All%':>6}")
        print(f"  {'-'*46}")
        for t in targets:
            fov = t * iso
            px = np.mean(col(rows, "fov_x") <= fov) * 100
            py = np.mean(col(rows, "fov_y") <= fov) * 100
            pz = np.mean(col(rows, "fov_z") <= fov) * 100
            # all three fit
            all_fit = np.mean(
                (col(rows, "fov_x") <= fov) &
                (col(rows, "fov_y") <= fov) &
                (col(rows, "fov_z") <= fov)
            ) * 100
            print(f"  {t:>7}  {fov:>8.0f}  {px:>5.1f}  {py:>5.1f}  {pz:>5.1f}  {all_fit:>6.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load(CSV_PATH)
    per_class_summary(rows)
    spacing_anisotropy(rows)
    coverage_table(rows)
    all_sims = simulate_all(rows)
    plots(rows)
    print(f"\nAll plots saved to {PLOT_DIR}/")
