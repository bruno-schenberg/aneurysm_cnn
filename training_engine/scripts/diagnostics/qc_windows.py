# D9.1 window sheets (ours and Lausanne), D9.4 clipped-window rates, D9.3 extra crops, D2.1 borderline Lausanne MIPs.
# Usage (from training_engine/): PYTHONPATH=$PWD .venv/bin/python scripts/diagnostics/qc_windows.py OUT_DIR
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib, numpy as np, pandas as pd
from config import DATASET_CSV

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); D = Path("/mnt/data/datasets")
WINDOWS = ("center90", "top60", "top75", "top90mm", "top120", "vessel60", "vessel90", "pct90", "full")
BORDERLINE = (107, 134, 195, 311, 325, 327, 410, 428)


def find(root, exam, ext):
    for c in ("0", "1"):
        p = root / c / f"{exam}{ext}"
        if p.exists(): return p
    raise FileNotFoundError(root / exam)


def mip(family, variant, exam): return np.load(find(D / family / variant, exam, ".npy")).astype(np.float32)
def show(ax, im, title):
    ax.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.8), aspect="auto"); ax.set_title(title, fontsize=6); ax.axis("off")


def load_windows(family):
    # windows.csv is appended to, so a re-generated exam can appear twice
    return pd.read_csv(D / family / "windows.csv").drop_duplicates(["exam", "window"], keep="last")


def content_check(family, w):
    """Correlate each stored window's sagittal MIP with the same Z range of the full-slab MIP, every exam."""
    rs = []
    for exam, g in w.groupby("exam"):
        full = mip(family, "full_stretch_512", exam)[2]; n = full.shape[1]
        for r in g.itertuples():
            c0, c1 = round(r.z_start_mm / r.extent_mm * n), round(r.z_end_mm / r.extent_mm * n)
            idx = np.linspace(0, n - 1, c1 - c0).round().astype(int)
            stored = mip(family, f"{r.window}_stretch_512", exam)[2][:, idx]
            rs.append((exam, r.window, np.corrcoef(stored.ravel(), full[:, c0:c1].ravel())[0, 1]))
    return pd.DataFrame(rs, columns=["exam", "window", "r"])


def window_sheets(family, exams, name, w):
    w = w.set_index(["exam", "window"])
    bands, bx = plt.subplots(len(exams), 9, figsize=(13, 1.5 * len(exams)))
    axial, ax = plt.subplots(len(exams), 9, figsize=(13, 1.5 * len(exams)))
    for i, (exam, label) in enumerate(exams):
        sag = np.rot90(mip(family, "full_stretch_512", exam)[2]); n = sag.shape[0]  # rot90: vertex up, anterior left
        for j, win in enumerate(WINDOWS):
            r = w.loc[(exam, win)]
            tag = f"{exam[:22]} {label}\n{win} {r.z_start_mm:.0f}-{r.z_end_mm:.0f}/{r.extent_mm:.0f}{' CLIP' if r.clipped else ''}"
            show(bx[i, j], sag, tag)
            top, bottom = n * (1 - r.z_end_mm / r.extent_mm), n * (1 - r.z_start_mm / r.extent_mm)
            bx[i, j].axhspan(0, top, color="k", alpha=.55); bx[i, j].axhspan(bottom, n, color="k", alpha=.55)
            bx[i, j].axhline(top, color="y", lw=.6); bx[i, j].axhline(bottom, color="y", lw=.6)
            show(ax[i, j], mip(family, f"{win}_stretch_512", exam)[0].T, tag)  # transpose: anterior up
    for f, s in ((bands, "bands"), (axial, "axial")):
        f.tight_layout(); f.savefig(OUT / f"qc_windows_{name}_{s}.png", dpi=110); plt.close(f)


df = pd.read_csv(DATASET_CSV).dropna(subset=["class"])
ours, laus = load_windows("mip_grid"), load_windows("mip_grid_lausanne")
df = df[df.exam.isin(set(ours.exam))]
def pick(sub, n, seed=0): return sub.sample(min(n, len(sub)), random_state=seed)

# D9.4
rates = pd.DataFrame({"ours": ours.groupby("window").clipped.mean(), "lausanne": laus.groupby("window").clipped.mean()}).round(3)
print("clipped rate per window\n", rates.loc[list(WINDOWS)].to_string())
print("extent mm, ours:\n", ours.groupby("exam").extent_mm.first().describe().round(1).to_string())
print("extent mm, Lausanne:\n", laus.groupby("exam").extent_mm.first().describe().round(1).to_string())
rates.to_csv(OUT / "clipped_rates.csv")

# D9.1
ex = pd.concat([pick(df[df.scanner == s], n) for s, n in (("skyra", 4), ("signa_hdxt", 3), ("optima_mr450w", 3))])
window_sheets("mip_grid", [(r.exam, f"{r.scanner} c{int(r._2)}") for r in ex.itertuples()], "ours", ours)
ls = pd.Series(sorted(set(laus.exam))).sample(10, random_state=0)
window_sheets("mip_grid_lausanne", [(e, "") for e in ls], "lausanne", laus)
for family, w in (("mip_grid", ours), ("mip_grid_lausanne", laus)):
    r = content_check(family, w); r.to_csv(OUT / f"content_check_{family}.csv", index=False)
    print(f"{family}: stored window vs full-slab crop, r per window\n", r.groupby("window").r.agg(["min", "median"]).round(4).to_string())
    print("worst:\n", r.nsmallest(5, "r").to_string(index=False))

# D9.3: more crops, seed 1 (the first sheet used seed 0)
ex = pick(df, 60, 1)
fig, ax = plt.subplots(10, 12, figsize=(13, 12))
for k, r in enumerate(ex.itertuples()):
    c = nib.load(find(D / "vessel_crop_256", r.exam, ".nii.gz")).get_fdata(dtype=np.float32)
    show(ax[k // 6, 2 * (k % 6)], np.rot90(c.max(2)), f"{r.exam} ax")
    show(ax[k // 6, 2 * (k % 6) + 1], np.rot90(c.max(0)), f"{r.exam} sag")
fig.tight_layout(); fig.savefig(OUT / "qc_vessel_crop_more.png", dpi=110); plt.close(fig)

# D2.1: every session of the borderline subjects, with the share of Z slices that hold signal
exams = sorted(e for e in set(laus.exam) if int(e.split("_")[1].removeprefix("sub-")) in BORDERLINE)
fig, ax = plt.subplots(len(exams), 3, figsize=(8, 2.7 * len(exams)))
for i, e in enumerate(exams):
    v = nib.load(find(Path("/mnt/data/lausanne_nifti"), e.removeprefix("laus_"), ".nii.gz")).get_fdata(dtype=np.float32)
    frac = (v > 0).any((0, 1)).mean(); m = mip("mip_grid_lausanne", "full_pad_512", e)
    print(f"{e}: z slices with signal {frac:.2f}, shape {v.shape}, extent {laus[laus.exam == e].extent_mm.iloc[0]} mm")
    for j, (im, t) in enumerate(((m[0].T, "axial"), (np.rot90(m[1]), "coronal"), (np.rot90(m[2]), "sagittal"))):
        show(ax[i, j], im, f"{e} signal {frac:.2f} {t}")
fig.tight_layout(); fig.savefig(OUT / "qc_lausanne_borderline.png", dpi=90); plt.close(fig)
