"""Thesis figures from saved predictions; labels in Portuguese, as in the thesis.

Test ROC: all curves come from ensemble_out/test_percentiles.csv, so the 2D curve is seed 42,
the member actually in the ensemble (0.629), not the fold x seed average (0.685).

MIP windows: one dev exam, each Z window on the full-slab sagittal MIP next to its axial MIP.

MIL attention (2D-4_mil_tiles): 2 true positives and 2 false negatives drawn at random among dev
positives of one territory, each scored by the fold model that held it out. Tiles span the three
stacked channels, so one attention value covers the same tile position in all three views.

Usage (from training_engine/): python -m analysis.thesis_figures [--out thesis_stats_out]
"""
import argparse
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.bootstrap_auc import boot_ci
from analysis.predict import exam_id, predict
from config import DATASET_CSV

# Line style repeats the identity for grayscale print and colour-blind readers
MODELS = (("ensemble", "Ensemble (2D + 3D + nnU-Net)", "#2a78d6", "-", 2.4),
          ("nnunet", "nnU-Net (sem treino nos nossos dados)", "#eda100", (0, (7, 2)), 1.8),
          ("mip", "2D (ResNet50, fusão tardia)", "#eb6834", "--", 1.8),
          ("r3d_iso", "3D (R3D18, iso_256, idade e sexo)", "#1baf7a", "-.", 1.8))


MIP_ROOT = Path("/mnt/data/datasets/mip_grid")
ROOT = Path(__file__).resolve().parents[2]
WINDOWS = ("center90", "top60", "top75", "top90mm", "top120", "vessel60", "vessel90", "pct90", "full")
PANEL_EXAM, PANEL_CLASS = "BP0233", 0  # dev, negative, Skyra: the most common scanner
MIL = Path("experiments/2D-4_mil_tiles")
TERRITORY = {"ICA": "ACI", "MCA": "ACM", "ACA": "ACA"}
ATTENTION_CMAP = LinearSegmentedColormap.from_list("att", [(.92, .41, .20, 0), (.92, .41, .20, .85)])


def comma(x, digits=3): return f"{x:.{digits}f}".replace(".", ",")


def test_roc(out):
    t = pd.read_csv("ensemble_out/test_percentiles.csv", index_col=0)
    y = t.true_label.to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color="#898781", lw=.8, ls=":")
    for col, name, color, ls, lw in MODELS:
        fpr, tpr, _ = roc_curve(y, t[col])
        auc, (lo, hi) = roc_auc_score(y, t[col]), boot_ci(y, t[col].to_numpy())
        print(f"{col}: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.plot(fpr, tpr, color=color, ls=ls, lw=lw, solid_capstyle="round",
                label=f"{name}\nAUC {comma(auc)} [{comma(lo)}; {comma(hi)}]")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), aspect="equal", xlabel="1 − especificidade", ylabel="Sensibilidade")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 1)))
    ax.set_xticks(np.arange(0, 1.01, .2)); ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(color="#e1e0d9", lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0, labelsize=9)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor="white", edgecolor="none", framealpha=.92, labelspacing=.7, handlelength=3.2)
    fig.tight_layout(); fig.savefig(out / "roc_teste.png", dpi=300, facecolor="white"); plt.close(fig)


def gray(ax, im):
    ax.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.8), aspect="auto", interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def windows_panel(out):
    assert PANEL_EXAM not in json.loads((ROOT / "dataset/test_ids.json").read_text())
    w = pd.read_csv(MIP_ROOT / "windows.csv").drop_duplicates(["exam", "window"], keep="last")
    w = w[w.exam == PANEL_EXAM].set_index("window")
    load = lambda variant: np.load(MIP_ROOT / variant / str(PANEL_CLASS) / f"{PANEL_EXAM}.npy").astype(np.float32)
    sag = np.rot90(load("full_stretch_512")[2]); n = sag.shape[0]  # vertex up, anterior left
    fig, ax = plt.subplots(3, 6, figsize=(6.6, 3.9))
    for k, win in enumerate(WINDOWS):
        r = w.loc[win]; a, b = ax[k // 3, 2 * (k % 3)], ax[k // 3, 2 * (k % 3) + 1]
        gray(a, sag); a.set_box_aspect(r.extent_mm / 180)
        top, bottom = n * (1 - r.z_end_mm / r.extent_mm), n * (1 - r.z_start_mm / r.extent_mm)
        a.axhspan(-.5, top, color="k", alpha=.62); a.axhspan(bottom, n - .5, color="k", alpha=.62)
        a.axhline(top, color="#eda100", lw=.9); a.axhline(bottom, color="#eda100", lw=.9)
        gray(b, load(f"{win}_stretch_512")[0].T); b.set_box_aspect(1)  # anterior up
        final = win == "vessel90"
        a.set_title(f"{win}{' (modelo final)' if final else ''}\n{r.z_start_mm:.0f}–{r.z_end_mm:.0f} mm de {r.extent_mm:.0f}",
                    fontsize=7, loc="left", fontweight="bold" if final else "normal", color="#0b0b0b")
    fig.tight_layout(w_pad=.3, h_pad=.6); fig.savefig(out / "janelas_mip.png", dpi=300, facecolor="white"); plt.close(fig)


def mil_attention(out, device):
    oof = pd.concat(pd.read_csv(d / "predictions.csv").assign(fold=d.name) for d in sorted(MIL.glob("fold_*")))
    oof["exam"] = oof.image.map(exam_id)
    location = pd.read_csv(DATASET_CSV).set_index("exam").location
    pos = oof[(oof.true_label == 1) & oof.exam.map(location).isin(TERRITORY)]
    tp, fn = pos[pos.prob >= .5], pos[pos.prob < .5]
    print(f"dev positives of one territory: {len(pos)} ({len(tp)} TP, {len(fn)} FN at 0.5)")
    picks = pd.concat([tp.sample(2, random_state=0).assign(kind="VP"), fn.sample(2, random_state=0).assign(kind="FN")])
    att_dir = out / "mil_attention"
    pred = predict(MIL, ids=set(picks.exam), device=device, attention_dir=att_dir).set_index("exam_id")
    for r in picks.itertuples():  # the rerun must reproduce the score saved by the training run
        assert abs(pred.loc[r.exam, f"prob_{r.fold}"] - r.prob) < .02, (r.exam, r.prob, pred.loc[r.exam, f"prob_{r.fold}"])
    maps = {r.exam: np.load(att_dir / f"{r.exam}_{r.fold}.npy") for r in picks.itertuples()}
    vmax = max(m.max() for m in maps.values())
    views = ((lambda a: a[0].T, "Axial"), (lambda a: np.rot90(a[1]), "Coronal"), (lambda a: np.rot90(a[2]), "Sagital"))
    fig, ax = plt.subplots(4, 3, figsize=(5.4, 7.2), layout="constrained")
    for i, r in enumerate(picks.itertuples()):
        mip = np.load(r.image).astype(np.float32); m = maps[r.exam]
        scale = mip.shape[1] // m.shape[0]; assert m.shape[0] * scale == mip.shape[1]
        heat = np.repeat(np.repeat(m, scale, 0), scale, 1)[None].repeat(3, 0)
        for j, (view, name) in enumerate(views):
            gray(ax[i, j], view(mip)); ax[i, j].set_box_aspect(1)
            shown = ax[i, j].imshow(view(heat), cmap=ATTENTION_CMAP, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
            if i == 0:
                ax[i, j].set_title(name, fontsize=9, color="#0b0b0b")
        ax[i, 0].set_ylabel(f"{r.kind} · {TERRITORY[location[r.exam]]}\np = {comma(r.prob, 2)}", fontsize=8.5, color="#0b0b0b")
        print(r.kind, r.exam, location[r.exam], r.fold, f"prob {r.prob:.3f}, max attention {m.max():.3f} of {m.size} tiles")
    bar = fig.colorbar(shown, ax=ax, orientation="horizontal", fraction=.03, pad=.01, aspect=40)
    bar.set_label("Atenção por bloco", fontsize=8, color="#52514e"); bar.ax.tick_params(labelsize=7, colors="#52514e", length=0)
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 2))); bar.outline.set_visible(False)
    fig.savefig(out / "mil_atencao.png", dpi=300, facecolor="white"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("thesis_stats_out"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    test_roc(a.out)
    windows_panel(a.out)
    mil_attention(a.out, a.device)


if __name__ == "__main__":
    main()
