"""Step 3 — Visual QC: render pipeline stages side-by-side.

Shows one exam per resolution group at each pipeline stage:
DICOM | NIfTI | iso_128 | iso_256 (only stages that exist on disk).

Output: reports/step3_visualize.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from medprep.visualize import load_mid_slice_dicom, load_mid_slice_nifti


def _nifti_panel(path: Path) -> tuple[np.ndarray, str]:
    """Return (2D slice, info string) for the middle axial slice of a NIfTI."""
    slc, sp, shape = load_mid_slice_nifti(path)
    shape_str = f"{shape[0]}x{shape[1]}x{shape[2]}"
    sp_str    = f"{float(sp[0]):.3f}x{float(sp[1]):.3f}x{float(sp[2]):.3f}"
    info      = f"{shape_str}\n{sp_str} mm"
    return slc, info


def _dicom_panel(dcm_dir: Path, selected_series_uid: str | None = None) -> tuple[np.ndarray, str]:
    """Return (2D slice, info string) for the middle slice of a DICOM series."""
    slc, meta = load_mid_slice_dicom(dcm_dir, selected_series_uid)
    if not meta:
        return slc, "no valid slices"
    info = (f"{meta['columns']}x{meta['rows']}x{meta['n_slices']}\n"
            f"{meta['spacing_col']:.3f}x{meta['spacing_row']:.3f} mm (in-plane)")
    return slc, info


def _resolve_stages() -> list[str]:
    """Return stage names based on what variant dirs exist on disk."""
    stages = ["DICOM", "NIfTI"]
    for v in config.VARIANTS:
        vdir = config.DATASETS_ROOT / v["name"]
        if vdir.exists():
            stages.append(v["name"])
    return stages


def _select_exams(df: pd.DataFrame, per_group: int = 1) -> pd.DataFrame:
    """Pick per_group converted exams per resolution group."""
    converted = df[df["conversion_status"] == "converted"].copy()
    pieces = []
    for gid in sorted(converted["res_group"].dropna().unique()):
        gdf = converted[converted["res_group"] == int(gid)]
        pieces.append(gdf.head(per_group))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces).sort_values(["res_group", "exam_id"]).reset_index(drop=True)


def _stage_path(stage: str, exam_id: str, cls: str, dcm_path: Path) -> Path:
    """Resolve the on-disk path for one (stage, exam) cell of the grid."""
    if stage == "DICOM":
        return dcm_path
    if stage == "NIfTI":
        return config.NIFTI_ROOT / cls / f"{exam_id}.nii.gz"
    return config.DATASETS_ROOT / stage / cls / f"{exam_id}.nii.gz"


def _render(exams: pd.DataFrame, stages: list[str], out_path: Path) -> None:
    """Render the multi-stage comparison figure."""
    n_rows = len(exams)
    n_cols = len(stages)
    panel_w, panel_h = 4.0, 4.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_w, n_rows * panel_h),
        squeeze=False,
    )
    fig.patch.set_facecolor("black")
    fig.suptitle(
        "Pipeline stages: " + " -> ".join(stages),
        fontsize=10, color="white", family="monospace", y=1.005,
    )

    for c, stage in enumerate(stages):
        axes[0, c].set_title(stage, fontsize=9, color="cyan", family="monospace", pad=8)

    for r, (_, row) in enumerate(exams.iterrows()):
        exam_id  = str(row["exam_id"])
        cls      = str(int(row["class"]))
        dcm_path = Path(str(row["dcm_path"]))
        series_uid_raw = row.get("selected_series_uid")
        series_uid = str(series_uid_raw) if pd.notna(series_uid_raw) else None

        gid       = int(row.get("res_group", 0))
        mtx       = f"{int(row.get('rows', 0))}x{int(row.get('columns', 0))}"
        sx        = float(row.get("voxel_spacing_row", 0))
        sz        = float(row.get("voxel_spacing_z", 0))
        cls_label = "aneurysm" if cls == "1" else "control"
        row_label = f"G{gid} {exam_id}\n{mtx} sx={sx:.3f} sz={sz:.3f}\n[{cls_label}]"
        cls_col   = "#ff6666" if cls == "1" else "#aaaaaa"

        for c, stage in enumerate(stages):
            ax = axes[r, c]
            ax.set_facecolor("black")
            ax.axis("off")

            path = _stage_path(stage, exam_id, cls, dcm_path)

            if not path.exists():
                ax.text(0.5, 0.5, f"missing\n{path.name}",
                        ha="center", va="center", color="red", fontsize=7,
                        transform=ax.transAxes)
                continue

            try:
                if stage == "DICOM":
                    slc, info = _dicom_panel(path, series_uid)
                else:
                    slc, info = _nifti_panel(path)
            except Exception as e:
                ax.text(0.5, 0.5, f"error\n{e}",
                        ha="center", va="center", color="red", fontsize=6,
                        transform=ax.transAxes, wrap=True)
                continue

            ax.imshow(slc, cmap="gray", origin="upper", interpolation="bilinear")
            ax.text(0.02, 0.02, info,
                    ha="left", va="bottom", color="lime", fontsize=6,
                    family="monospace", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        axes[r, 0].text(-0.05, 0.5, row_label,
                        ha="right", va="center", color=cls_col, fontsize=8,
                        family="monospace", rotation=90,
                        transform=axes[r, 0].transAxes)

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def run(per_group: int = 1):
    if not config.STEP1_REPORT.exists():
        print(f"  ERROR: {config.STEP1_REPORT} not found. Run step 1 first.")
        sys.exit(1)

    df = pd.read_csv(config.STEP1_REPORT)
    selected = _select_exams(df, per_group=per_group)
    if selected.empty:
        print("[step3] No converted exams found. Skipping visualization.")
        return

    stages = _resolve_stages()
    print(f"[step3] Rendering {len(selected)} exam(s), stages: {' -> '.join(stages)}")

    for gid, gdf in selected.groupby("res_group"):
        r0 = gdf.iloc[0]
        print(f"  Group {int(gid)}: {', '.join(gdf['exam_id'].tolist())}  "
              f"({int(r0['rows'])}x{int(r0['columns'])} "
              f"sx={r0['voxel_spacing_row']:.4f} sz={r0['voxel_spacing_z']:.4f})")

    out_path = config.REPORTS_DIR / "step3_visualize.png"
    _render(selected, stages, out_path)
    print(f"\nStep 3 figure -> {out_path}  ({len(selected)} exams)")
