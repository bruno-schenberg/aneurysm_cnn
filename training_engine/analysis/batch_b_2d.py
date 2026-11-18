"""Thesis batch B (mestrado LACUNAS_RESULTADOS.md §B): B6 WCF and B5 scanner-stratified folds
vs the frozen 2D-3_512 (oversampling, label-only folds), 3 seeds each.

Test score per arm = mean prob of its 15 fold x seed models on the 114 test exams (as in
test_comparison_2d.py); paired bootstrap of the AUC difference, pooled and per seed.
Also prints per-fold val AUC and scanner mix, to tell fold noise from composition effects.

Usage (from training_engine/): python -m analysis.batch_b_2d
"""
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.bootstrap_auc import boot_ci, paired_boot_diff
from src.split import exam_key

EXP = Path("experiments")
SEEDS = ["", "_s1", "_s2"]

def test_probs(base, seeds=SEEDS):
    cols = [pd.read_csv(f).assign(e=lambda d: d.image.map(exam_key)).set_index("e").prob
            for s in seeds for f in sorted(glob.glob(str(EXP / f"{base}{s}/fold_*/test_predictions.csv")))]
    y = pd.read_csv(glob.glob(str(EXP / f"{base}{seeds[0]}/fold_1/test_predictions.csv"))[0]).assign(e=lambda d: d.image.map(exam_key)).set_index("e").true_label
    return y, pd.concat(cols, axis=1).mean(axis=1), len(cols)

y0, ref, n0 = test_probs("2D-3_512")
print(f"test n={len(y0)} pos={int(y0.sum())}; oversampling label-only: AUC {roc_auc_score(y0, ref):.3f} {np.round(boot_ci(y0.to_numpy(), ref.to_numpy()),3)} ({n0} models)")
for arm in ("2D-3_512_wcf", "2D-3_512_scanstrat"):
    y, p, n = test_probs(arm)
    p = p.reindex(ref.index); assert (y.reindex(ref.index) == y0).all()
    d, lo, hi = paired_boot_diff(y0.to_numpy(), p.to_numpy(), ref.to_numpy())
    print(f"{arm}: AUC {roc_auc_score(y0, p):.3f} ({n} models)  diff vs ref {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for s in SEEDS:  # per-seed, 5-fold ensemble each
        _, pa, _ = test_probs(arm, [s]); _, pb, _ = test_probs("2D-3_512", [s])
        d, lo, hi = paired_boot_diff(y0.to_numpy(), pa.reindex(ref.index).to_numpy(), pb.reindex(ref.index).to_numpy())
        print(f"   seed{s or '_42'}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

scan = {r["exam"].strip(): (r["brand"] + r["field_strength_T"]) or "unknown"
        for r in csv.DictReader(open(Path("../dataset/dataset.csv")))}
for base in ("2D-3_512", "2D-3_512_scanstrat"):
    print(f"\n{base} (seed 42) per fold: val AUC, n, pos rate, GE 1.5T share, GE 3T share")
    for f in sorted(glob.glob(str(EXP / f"{base}/fold_*/predictions.csv"))):
        v = pd.read_csv(f); e = v.image.map(exam_key); s = e.map(scan)
        print(f"  {Path(f).parent.name}: {roc_auc_score(v.true_label, v.prob):.3f}  n={len(v)}  pos={v.true_label.mean():.3f}  ge1.5={(s=='ge1.5').mean():.3f}  ge3={(s=='ge3.0').mean():.3f}")
