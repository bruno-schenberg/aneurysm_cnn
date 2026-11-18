"""Thesis statistics from saved predictions: operating point, per-location results, paired bootstraps.

Operating point, rule fixed before looking at test: per score, the threshold is the highest
one that keeps dev (OOF) sensitivity >= 90 %, and it is applied unchanged to the test exams.
The ensemble's dev score is the nested OOF one (ensemble_out, fitted on the other folds).
nnU-Net also gets its natural rule, "any positive voxel". The ensemble also gets a labelled
secondary point, max F2 on dev (ens@maxF2), added after the 90 % results were known; the
90 % rule stays the primary whichever scores better.

Per-location: positives of each territory against all negatives. Exams with more than one
territory form their own group; "missing" is left out.

Paired bootstraps: A - B per thesis claim, using only the seeds both runs have. Dev is the
pooled OOF of fold-wise percentiles, averaged over seeds; test is the fold x seed mean
probability, as in analysis/test_comparison_2d.py. No multiple-comparison correction.

Usage (from training_engine/): python -m analysis.thesis_stats [--out thesis_stats_out]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from analysis.bootstrap_auc import boot_ci, paired_boot_diff
from analysis.ensemble import load_member
from analysis.splits import label_only_folds
from analysis.test_comparison_2d import test_probs

ROOT = Path(__file__).resolve().parents[2]
EXP = Path("experiments")
SEG = ROOT / "segmentation/evaluation_c90z_384/predictions.csv"
SEEDS = ("", "_s1", "_s2")
BASE = "2D-2_latefusion_vessel90"  # the 384 late-fusion run stages 3-6 were derived from
EARLY = "2D-1_vessel90"  # same 3-channel input as Swin-T and the MIL tiles
FINAL_2D = "2D-3_512"
PAIRS = [
    ("multimodal 2D", "2D-5_multimodal", BASE),
    ("multimodal 3D", "3D-1_r3d18k_iso256_mm", "3D-2_r3d18k_iso256"),
    ("Lausanne in training", "2D-6_mixed", BASE),
    ("window: vessel90 vs top75", "2D-1_vessel90", "2D-1_top75"),
    ("window: vessel90 vs center90", "2D-1_vessel90", "2D-1_center90"),
    ("window: vessel90 vs full", "2D-1_vessel90", "2D-1_full"),
    ("2D vs 3D, same anatomy", FINAL_2D, "3D-5_r3d18k_vessel90z256"),
    ("2D vs 3D iso + age/sex", FINAL_2D, "3D-1_r3d18k_iso256_mm"),
    ("2D vs 3D aniso512", FINAL_2D, "3D-9_r3d18k_aniso512"),
    ("attention: Swin-T vs ResNet", "2D-4_swin_t", EARLY),
    ("attention: MIL tiles vs ResNet", "2D-4_mil_tiles", EARLY),
    ("N4, early fusion 384", "2D-1b_n4_vessel90", EARLY),
    ("N4, late fusion 384", "2D-8_n4_latefusion_384", BASE),
    ("N4, late fusion 512", "2D-8_n4_latefusion_512", FINAL_2D),
]
POSTERIOR = {"BASILAR", "PCA", "VERTEBRAL", "VERTEBRAL; BASILAR"}


def dev_threshold(y, score, target=0.90):
    """Highest threshold t with sensitivity(score >= t) >= target."""
    pos = np.sort(score[y == 1])[::-1]
    return pos[math.ceil(target * len(pos)) - 1]


def dev_max_f2_threshold(y, score):
    """Threshold t that maximises F2 of (score >= t); ties go to the lower t (more sensitive)."""
    def f2(t):
        pred = score >= t
        tp = (pred & (y == 1)).sum()
        return 5 * tp / (5 * tp + 4 * (~pred & (y == 1)).sum() + (pred & (y == 0)).sum())
    return max(np.unique(score), key=f2)


def location_group(loc):
    if pd.isna(loc) or loc == "missing":
        return None
    if loc in POSTERIOR:
        return "posterior"
    return "multiple" if ";" in loc else loc


def _rate(k, n):
    if n == 0:
        return "-"
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return f"{k / n:.3f} [{ci.low:.3f}, {ci.high:.3f}] ({k}/{n})"


def _confusion(y, pred):
    return int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())


def operating_point(dev, test):
    """dev/test: {name: bool predictions}, plus 'y'."""
    rows = []
    for name in dev:
        if name == "y":
            continue
        row = {"score": name}
        for split, d in (("dev", dev), ("test", test)):
            tp, fn, tn, fp = _confusion(d["y"], d[name])
            row |= {f"{split}_sens": _rate(tp, tp + fn), f"{split}_spec": _rate(tn, tn + fp), f"{split}_ppv": _rate(tp, tp + fp)}
        rows.append(row)
    return pd.DataFrame(rows)


def per_location(split, frame, preds):
    negatives = frame[frame.true_label == 0]
    rows = []
    for loc, g in frame[frame.true_label == 1].groupby("group"):
        row = {"split": split, "location": loc, "n_pos": len(g)}
        for name in preds:
            if name in ("nnunet@90", "ens@maxF2"):  # nnunet@90 flags every exam (zero-voxel tie); maxF2 is reported overall only
                continue
            row[f"sens_{name}"] = round(preds[name].loc[g.index].mean(), 3)
        for col in ("mip", "r3d_iso", "nnunet", "ens"):
            s = pd.concat([g[col], negatives[col]])
            row[f"auc_{col}"] = round(roc_auc_score([1] * len(g) + [0] * len(negatives), s), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _seeds_of(a, b):
    both = [s for s in SEEDS if (EXP / f"{a}{s}/summary.json").exists() and (EXP / f"{b}{s}/summary.json").exists()]
    assert both, f"no shared seed for {a} and {b}"
    return both


def paired(folds, labels, dev_ids, test_ids):
    dev_index, test_index = sorted(dev_ids), sorted(test_ids)
    y_dev = np.array([labels[e] for e in dev_index])
    y_test = np.array([labels[e] for e in test_index])

    def scores(run, seeds):
        dirs = [EXP / f"{run}{s}" for s in seeds]
        oof = pd.concat([load_member(d, folds)[0] for d in dirs], axis=1).mean(axis=1).loc[dev_index].to_numpy()
        return oof, test_probs(dirs).p.loc[test_index].to_numpy()

    rows = []
    for claim, a, b in PAIRS:
        seeds = _seeds_of(a, b)
        (da, ta), (db, tb) = scores(a, seeds), scores(b, seeds)
        rows.append({"claim": claim, "A": a, "B": b, "seeds": len(seeds),
                     "dev_A": roc_auc_score(y_dev, da), "dev_B": roc_auc_score(y_dev, db),
                     **dict(zip(("dev_diff", "dev_lo", "dev_hi"), paired_boot_diff(y_dev, da, db))),
                     "test_A": roc_auc_score(y_test, ta), "test_B": roc_auc_score(y_test, tb),
                     **dict(zip(("test_diff", "test_lo", "test_hi"), paired_boot_diff(y_test, ta, tb)))})
        print(f"  done: {claim}", flush=True)
    return pd.DataFrame(rows).round(3)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("thesis_stats_out"))
    a = p.parse_args()

    oof = pd.read_csv("ensemble_out/oof_percentiles.csv", index_col=0).rename(columns={"ensemble_nested": "ens"})
    test = pd.read_csv("ensemble_out/test_percentiles.csv", index_col=0).rename(columns={"ensemble": "ens"})
    seg = pd.read_csv(SEG).set_index("exam_id")
    locations = pd.read_csv(ROOT / "dataset/dataset.csv").set_index("exam").location
    frames = {"dev": oof, "test": test}
    for f in frames.values():
        f["group"] = f.index.map(locations).map(location_group)

    preds = {"dev": {}, "test": {}}
    for col in ("ens", "mip", "r3d_iso", "nnunet"):
        t = dev_threshold(oof.true_label.to_numpy(), oof[col].to_numpy())
        for split, f in frames.items():
            preds[split][f"{col}@90"] = f[col] >= t
        print(f"threshold {col}: {t:.4f}")
    for split, f in frames.items():
        preds[split]["nnunet_any_voxel"] = seg.n_positive_voxels.loc[f.index] > 0
    t = dev_max_f2_threshold(oof.true_label.to_numpy(), oof.ens.to_numpy())
    print(f"threshold ens max F2: {t:.4f}")
    for split, f in frames.items():
        preds[split]["ens@maxF2"] = f.ens >= t

    op = operating_point(*[{"y": frames[s].true_label.to_numpy(), **{k: v.to_numpy() for k, v in preds[s].items()}}
                           for s in ("dev", "test")])
    loc = pd.concat(per_location(s, frames[s], preds[s]) for s in ("dev", "test"))

    labels = dict(zip(seg.index, seg.true_label))
    test_ids, folds = label_only_folds(labels)
    pairs = paired(folds, labels, set().union(*folds), test_ids)

    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", None)
    for name, df in (("operating_point", op), ("per_location", loc), ("paired_bootstraps", pairs)):
        df.to_csv(a.out / f"{name}.csv", index=False)
        print(f"\n{name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
