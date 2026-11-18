"""Aggregate all experiment results (completed, incomplete, not run) into one CSV.

Usage: python analysis/aggregate_results.py OUT.csv   (~3 min, mostly bootstrap CIs)
The not-run rows are hand-written.
"""
import csv, glob, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "training_engine/experiments"
OUT = Path(sys.argv[1])
sys.argv = sys.argv[:1]  # bootstrap_auc.py loops over argv at import
from bootstrap_auc import boot_ci  # same CI method as the thesis tables

COLS = ["date_start", "date_end", "experiment", "family", "phase", "status", "model", "dataset",
        "external_train_data", "eval_protocol", "eval_set", "n_eval", "n_pos_eval",
        "auc", "auc_std_folds", "auc_pooled_oof", "auc_ci95_low", "auc_ci95_high",
        "n_test", "n_pos_test", "auc_test_mean_folds", "auc_test_fold_ensemble",
        "auc_test_ci95_low", "auc_test_ci95_high",
        "precision", "recall", "specificity", "f2", "accuracy", "degenerate",
        "epochs_run", "lr", "batch_size", "grad_accum", "loss", "balancing", "checkpoint_metric",
        "freeze_epochs", "dropout", "augmentation", "amp", "synthetic_sphere", "notes", "source"]

def family(name, model):
    n = name.lower()
    if n.startswith("2d-"): return "MIP 2D classification"
    if n.startswith("3d-0_sphere_mil"): return "3D MIL"
    if n.startswith("3d-0_sphere_swin"): return "3D attention/transformer"
    if "modelnet" in n or n.startswith(("smoke", "test_submit")): return "sanity/smoke"
    if n.startswith("diag_") or n.startswith("monai256"): return "3D ROCm diagnostics"
    if "nnunet_enc" in n: return "nnU-Net encoder transfer"
    if "convlstm" in n: return "2D sequence (Conv-LSTM)"
    if "frangi" in n: return "MIP Frangi"
    if n.startswith(("mip", "p6")): return "MIP 2D classification"
    if n.startswith("mil_"): return "3D MIL"
    return "3D whole-volume classification"

def phase(name):
    m = re.match(r"([23]D-\d+b?)", name)
    if m: return m.group(1)
    n = name.lower()
    for p, k in [("6A", "p6a"), ("6B", "p6b"), ("6C", "p6c"), ("6D", "p6d"), ("6E", "p6e")]:
        if n.startswith(k): return p
    if "nnunet_enc" in n: return "4A"
    if "allaug_" in n and "kfold" not in n: return "3C"
    if "_aug_" in n or "allaug_kfold" in n: return "3B"
    return ""

NOTES = {
    "2D-6_lausanne_only": "val and test are Lausanne exams; not comparable with the other runs",
    "R3D18K_256_lr1e4_5439": "best 3D run",
    "R3D18K_128_oversample_5515": "best 3D at 128^3",
    "R3D18K_128_sphere_5546": "synthetic sphere r=2.5 injected",
    "R3D18K_128_sphere_r5_5546": "sanity check: obvious synthetic sphere r=5 is learnable",
    "MIP_earlyfusion_top90_kfold": "k-fold confirmed best MIP",
    "MIP_earlyfusion_mixed_v2_kfold": "k-fold, center crop + Lausanne in train",
    "MIP_earlyfusion_top90": "single split; fell to 0.712 under k-fold",
    "MIP_top90_aug_allaug": "single split; fell to 0.701 under k-fold",
    "P6D_mil_slice_2d": "failed to converge; killed by cluster shutdown at epoch 24",
    "P6E_vit_b16_top90": "ViT degenerate (recall=1, spec=0)",
    "ConvLSTM_c90z384": "ResNet18 per-slice + BiLSTM, degenerate all-positive",
}

def log_dates(d):
    logs = sorted(d.glob("*.log"))
    if not logs: return "", "", None
    lines = logs[0].read_text(errors="ignore").splitlines()
    ts = [l[:19] for l in lines if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", l)]
    epochs = [l for l in lines if re.search(r"Epoch \d+/\d+", l)]
    return (ts[0] if ts else ""), (ts[-1] if ts else ""), epochs

def aug(c):
    parts = [f"{k[4:].lower()}={c[k]}" for k in sorted(c) if k.startswith("AUG_") and k.endswith("_PROB") and c[k]]
    return "; ".join(parts)

rows = []
for d in sorted(p for p in EXP.iterdir() if p.is_dir()):
    cfg_f = d / "config.json"
    if not cfg_f.exists(): continue
    c = json.loads(cfg_f.read_text())
    start, end, epochs = log_dates(d)
    r = dict(experiment=d.name, date_start=start, date_end=end, model=c.get("model"),
             dataset=c.get("data_path_key"), external_train_data=c.get("EXTERNAL_DATA_PREFIX") or "",
             family=family(d.name, c.get("model")), phase=phase(d.name),
             lr=c.get("LEARNING_RATE"), batch_size=c.get("BATCH_SIZE"), grad_accum=c.get("GRAD_ACCUM_STEPS"),
             loss=c.get("LOSS_FN"), balancing=c.get("balancing"), checkpoint_metric=c.get("CHECKPOINT_METRIC"),
             freeze_epochs=c.get("FREEZE_BACKBONE_EPOCHS"), dropout=c.get("DROPOUT", ""),
             augmentation=aug(c), amp=c.get("AMP_DTYPE") if c.get("USE_AMP") else "off",
             synthetic_sphere=c.get("SYNTHETIC_SPHERE_RADIUS") if c.get("ADD_SYNTHETIC_SPHERE") else "",
             eval_protocol=f"{c.get('N_SPLITS')}-fold CV" if c.get("USE_KFOLD") else "single 80/20 split",
             eval_set="validation + test" if c.get("EVAL_ON_TEST") else "validation",
             notes=NOTES.get(d.name, ""), source=str(d.relative_to(ROOT)))
    if epochs:
        r["epochs_run"] = re.search(r"Epoch (\d+)/", epochs[-1]).group(1)
    s_f = d / "summary.json"
    if s_f.exists():
        s = json.loads(s_f.read_text())
        r["status"] = "completed"
        r["auc"] = round(s["mean_auc"], 4)
        if s["num_folds"] > 1: r["auc_std_folds"] = round(s["std_auc"], 4)
        m = pd.DataFrame(s["per_fold"]).T.mean()
        r.update(precision=round(m.Precision, 4), recall=round(m.Recall, 4),
                 specificity=round(m.Specificity, 4), f2=round(m["F2-Score"], 4), accuracy=round(m.Accuracy, 4))
        if (pd.DataFrame(s["per_fold"]).T[["Recall", "Specificity"]] == 0).any().any():
            r["degenerate"] = "yes (some fold predicts a single class)"
        if "test" in s:  # dlorchestra >= 0.2: val and test both scored
            r["auc_test_mean_folds"] = round(s["test"]["mean_auc"], 4)
            t = pd.concat(pd.read_csv(f) for f in sorted(d.glob("fold_*/test_predictions.csv")))
            t = t.groupby("image").agg(true_label=("true_label", "first"), prob=("prob", "mean"))
            yt, pt = t.true_label.to_numpy(), t.prob.to_numpy()
            lo, hi = boot_ci(yt, pt)
            r.update(n_test=len(yt), n_pos_test=int(yt.sum()), auc_test_fold_ensemble=round(roc_auc_score(yt, pt), 4),
                     auc_test_ci95_low=round(lo, 3), auc_test_ci95_high=round(hi, 3))
        preds = sorted(d.glob("fold_*/predictions.csv"))
        if preds:
            p = pd.concat(pd.read_csv(f) for f in preds)
            y, pr = p.true_label.to_numpy(), p.prob.to_numpy()
            r.update(n_eval=len(y), n_pos_eval=int(y.sum()))
            if y.min() != y.max():
                lo, hi = boot_ci(y, pr)
                r.update(auc_ci95_low=round(lo, 3), auc_ci95_high=round(hi, 3))
                if len(preds) > 1: r["auc_pooled_oof"] = round(roc_auc_score(y, pr), 4)
    else:
        last = epochs[-1] if epochs else ""
        va = re.search(r"val_AUC=([\d.]+)", last)
        r["status"] = "incomplete (no summary; crashed/killed)"
        if va: r["notes"] = (r["notes"] + "; " if r["notes"] else "") + f"last logged val_AUC={va.group(1)} (not final)"
    rows.append(r)

# nnU-Net zero-shot (no training) and the score-level ensemble
for seg_dir, dataset, note in [
        ("evaluation", "original NIfTI", ""),
        ("evaluation_c90z_384", "c90z_384",
         "best nnU-Net input; nnunetv2 2.8.1 rerun on Drummond (Sep-2026 run: 0.723)")]:
    seg = json.loads((ROOT / "segmentation" / seg_dir / "summary.json").read_text())
    p = pd.read_csv(ROOT / "segmentation" / seg_dir / "predictions.csv")
    lo, hi = boot_ci(p.true_label.to_numpy(), p.n_positive_voxels.to_numpy())
    rows.append(dict(experiment=f"nnUNet_zeroshot_{dataset.replace(' ', '_').lower()}",
                     family="nnU-Net zero-shot segmentation", status="completed",
                     model="nnU-Net multi-centric (Zenodo 13386859)", dataset=dataset,
                     eval_protocol="zero-shot (no training)", eval_set="all exams", n_eval=seg["n_samples"],
                     n_pos_eval=seg["tp"] + seg["fn"], auc=round(seg["auc_voxel_count"], 4),
                     auc_ci95_low=round(lo, 3), auc_ci95_high=round(hi, 3),
                     precision=round(seg["precision"], 4), recall=round(seg["sensitivity"], 4),
                     specificity=round(seg["specificity"], 4), f2=round(seg["f2"], 4),
                     accuracy=round((seg["tp"] + seg["tn"]) / seg["n_samples"], 4),
                     notes="; ".join(filter(None, ["score = predicted aneurysm voxel count", note])),
                     source=f"segmentation/{seg_dir}/summary.json"))
j = pd.read_csv(EXP / "phase5_error_analysis/joined_predictions.csv")
j = j.dropna(subset=["seg_prob", "mip_top90_prob", "mip_center_prob"])
ens = j[["seg_prob", "mip_top90_prob", "mip_center_prob"]].mean(axis=1).to_numpy()
y = j.true_label.to_numpy()
lo, hi = boot_ci(y, ens)
rows.append(dict(experiment="Phase5_ensemble_avg3", date_end="2026-10-22", family="Ensemble (score-level)",
                 phase="5", status="completed", model="mean(nnU-Net c90z_384, MIP top90 k-fold, MIP center+Lausanne k-fold)",
                 eval_protocol="pooled OOF + zero-shot", eval_set="validation (OOF)", n_eval=len(y),
                 n_pos_eval=int(y.sum()), auc=round(roc_auc_score(y, ens), 4),
                 auc_ci95_low=round(lo, 3), auc_ci95_high=round(hi, 3),
                 notes="best overall, zero extra training; paired bootstrap vs nnU-Net c90z_384 +0.037 [-0.007,+0.083] (not significant); union sensitivity 68.2%; rerun 2026-10-22 with c90z_384 (Sep-2026 run used original-NIfTI nnU-Net: 0.755)",
                 source="training_engine/experiments/phase5_error_analysis/joined_predictions.csv"))

# The final ensemble, chosen on nested OOF and scored once on test.
e = json.loads((ROOT / "training_engine/ensemble_out/summary.json").read_text())
chosen = e["rules_nested_oof"][e["chosen_rule"]]
rows.append(dict(experiment="E_final_ensemble", date_end="2026-11-08", family="Ensemble (score-level)", phase="E",
                 status="completed", model=f"{e['chosen_rule']}(2D-3_512, 3D-1_r3d18k_iso256_mm, nnU-Net c90z_384)",
                 eval_protocol="nested 5-fold OOF, test scored once", eval_set="validation + test",
                 n_eval=e["n_dev"], auc=round(chosen["auc"], 4), auc_ci95_low=round(chosen["ci"][0], 3),
                 auc_ci95_high=round(chosen["ci"][1], 3), n_test=e["n_test"], n_pos_test=e["n_test_pos"],
                 auc_test_fold_ensemble=round(e["test"]["ensemble"]["auc"], 4),
                 auc_test_ci95_low=round(e["test"]["ensemble"]["ci"][0], 3),
                 auc_test_ci95_high=round(e["test"]["ensemble"]["ci"][1], 3),
                 notes="percentile-rank stack", source="training_engine/ensemble_out/summary.json"))

# Runs that were never made.
_GATE = "cut: every backbone failed the sphere gate (val < 0.85)"
_SKIP = "never run: gated on the sphere test its backbone failed"
NOT_RUN = [
    ("3D-3_medicalnet_r18", "3D whole-volume classification", "3D-3", _GATE, "MedicalNet R18 iso_256 zscore"),
    ("3D-4_medicalnet_r50", "3D whole-volume classification", "3D-4", _SKIP, "MedicalNet R50 iso_256 zscore"),
    ("3D-7_swin_ssl_c90z256", "3D attention/transformer", "3D-7", _GATE, "Swin encoder + MONAI SSL weights, c90z_256"),
    ("3D-8_swin_ssl_c90z384", "3D attention/transformer", "3D-8", _SKIP, "Swin SSL c90z_384, single split"),
    ("3D-10_mil_r3d18k", "3D MIL", "3D-10", _GATE, "gated attention over 64^3 patches, R3D18K"),
    ("connectomicslab_zeroshot", "External model / validation", "2.7", "not run: optional", "3D U-Net, ADAM-only and ADAM->Lausanne weights"),
    ("nnunet_self_training_pilot", "nnU-Net fine-tuning", "4.6", "not run: optional", "fine-tune on empty negatives + single-candidate pseudo-labels"),
    ("Backlog_weak_labels_reports", "weak supervision", "backlog", "future work", "radiology reports -> 3D coordinates"),
]
for name, fam, ph, st, model in NOT_RUN:
    rows.append(dict(experiment=name, family=fam, phase=ph, status=st, model=model, source="not run"))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, COLS, extrasaction="raise")
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {OUT}")
