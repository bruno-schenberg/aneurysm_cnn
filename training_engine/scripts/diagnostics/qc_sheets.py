# D9.2 MIP orientation sheet and D9.3 vessel-crop sheet, for visual QC.
# Usage (from training_engine/): PYTHONPATH=$PWD .venv/bin/python scripts/diagnostics/qc_sheets.py OUT_DIR
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib, numpy as np, pandas as pd
from config import DATASET_CSV
OUT = Path(sys.argv[1]); D = Path("/mnt/data/datasets")
df = pd.read_csv(DATASET_CSV).dropna(subset=["class"])
have = lambda root, ext: {f.name.removesuffix(ext) for f in (D / root).glob("*/*" + ext)}
df = df[df.exam.isin(have("mip_grid/vessel90_stretch_512", ".npy") & have("vessel_crop_256", ".nii.gz") & have("iso_256", ".nii.gz"))]
def pick(sub, n): return sub.sample(min(n, len(sub)), random_state=0)

def find(root, exam, ext):
    for c in ("0", "1"):
        p = root / c / f"{exam}{ext}"
        if p.exists(): return p

# D9.2: 10 exams across scanners, 3 MIP channels of the final 2D input.
ex = pd.concat([pick(df[df.scanner == s], n) for s, n in (("skyra", 4), ("signa_hdxt", 3), ("optima_mr450w", 3))])
fig, ax = plt.subplots(len(ex), 3, figsize=(9, 3 * len(ex)))
for i, r in enumerate(ex.itertuples()):
    m = np.load(find(D / "mip_grid/vessel90_stretch_512", r.exam, ".npy")).astype(np.float32)
    for j, t in enumerate(("axial", "coronal", "sagittal")):
        ax[i, j].imshow(m[j], cmap="gray"); ax[i, j].set_title(f"{r.exam} {r.scanner} — ch{j} '{t}'", fontsize=8); ax[i, j].axis("off")
fig.tight_layout(); fig.savefig(OUT / "qc_mip_orientation.png", dpi=70); plt.close(fig)

# D9.3: vessel_crop_256 MIPs next to the full iso_256 volume, 20 exams; posterior-circulation positives first.
post = df[df.location.fillna("").str.contains("BASILAR|VERTEBRAL|PCA")]
pos = pd.concat([post, pick(df[(df["class"] == 1) & ~df.index.isin(post.index)], 12 - len(post))])
ex = pd.concat([pos, pick(df[df["class"] == 0], 8)])
fig, ax = plt.subplots(len(ex), 4, figsize=(12, 3 * len(ex)))
for i, r in enumerate(ex.itertuples()):
    c = nib.load(find(D / "vessel_crop_256", r.exam, ".nii.gz")).get_fdata(dtype=np.float32)
    f = nib.load(find(D / "iso_256", r.exam, ".nii.gz")).get_fdata(dtype=np.float32)
    ims = [c.max(2), c.max(1), c.max(0), f.max(0)]
    names = ["crop: max over axis 2", "crop: max over axis 1", "crop: max over axis 0", "FULL iso_256: max over axis 0"]
    for j in range(4):
        ax[i, j].imshow(np.rot90(ims[j]), cmap="gray")
        ax[i, j].set_title(f"{r.exam} cls={int(r._2)} {r.location if isinstance(r.location, str) else ''} — {names[j]}", fontsize=7)
        ax[i, j].axis("off")
fig.tight_layout(); fig.savefig(OUT / "qc_vessel_crop.png", dpi=60); plt.close(fig)
print(ex[["exam", "class", "location", "scanner"]].to_string(index=False))
