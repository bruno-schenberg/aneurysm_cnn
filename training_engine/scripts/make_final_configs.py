"""Write configs/final/<ID>.json for the final campaign.

Stages, each run after the previous one's experiments have finished:
  (default)  2D-1 window grid + all 3D configs
  --stage 2  2D-1b (preprocessing) and 2D-2 (views/aspect) for the top-2
             2D-1 windows
  --stage 3  2D-3..2D-6 on the best 2D config so far
"Best" = highest mean val AUC over folds (summary.json "mean_auc"), never test.
2D-1/1b/2 run with SEEDS (training seed only; folds and test stay fixed) and
are ranked on the mean over seeds: single-seed differences were within noise.
That also covers seed variance, so there is no separate 2D-7.

Usage:
    python scripts/make_final_configs.py
    python scripts/make_final_configs.py --stage 2
    python scripts/make_final_configs.py --stage 3
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

OUT = ROOT / "configs" / "final"

BASE = {  # campaign rules
    "USE_KFOLD": True, "N_SPLITS": 5, "STRATIFY_BY_SCANNER": False,
    "CHECKPOINT_METRIC": "val_auc", "balancing": "oversampling",
    "EVAL_ON_TEST": True, "TEST_IDS_FILE": "dataset/test_ids.json",
    "EXCLUSIONS_FILE": "dataset/exclusions.json",
    "RANDOM_SEED": 42, "SPLIT_SEED": 42,
}
MIP = {  # MIP_earlyfusion_top90_kfold (0.712 OOF)
    **BASE, "model": "RESNET50_2D_3CH", "SPATIAL_DIMS": 2, "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4, "WEIGHT_DECAY": 1e-3, "EPOCHS": 100,
    "FREEZE_BACKBONE_EPOCHS": 5, "USE_AMP": True, "AMP_DTYPE": "bfloat16",
}
R3D = {  # R3D18K_256_lr1e4 (best 3D); bf16 on CUDA (fp16 was a ROCm workaround)
    **BASE, "model": "R3D18_KINETICS", "data_path_key": "iso_256", "EPOCHS": 80,
    "BATCH_SIZE": 4, "GRAD_ACCUM_STEPS": 4, "LEARNING_RATE": 1e-4, "WEIGHT_DECAY": 1e-4,
    "COSINE_T0": 80, "COSINE_ETA_MIN": 1e-7, "EARLY_STOPPING_PATIENCE": 25,
    "FREEZE_BACKBONE_EPOCHS": 5, "normalization": "percentile", "NUM_WORKERS": 4,
    "USE_AMP": True, "AMP_DTYPE": "bfloat16", "CUDNN_BENCHMARK": True,
}
MEDNET = {"normalization": "zscore"}  # MedicalNet's own preprocessing (tasks 1.6)
SWIN = {"model": "SWIN_ENC", "BATCH_SIZE": 1, "GRAD_ACCUM_STEPS": 16, "DROPOUT": 0.5,
        "WARMUP_EPOCHS": 5, "EPOCHS": 50, "EARLY_STOPPING_PATIENCE": 15}
SINGLE = {"USE_KFOLD": False}
SPHERE = {**SINGLE, "ADD_SYNTHETIC_SPHERE": True, "SYNTHETIC_SPHERE_RADIUS": 2.5}
SEEDS = (42, 1, 2)  # 42 = the base name, others get a _s<seed> suffix


def seeded(configs: dict[str, dict]) -> dict[str, dict]:
    return {f"{n}_s{s}" if s != 42 else n: {**c, "RANDOM_SEED": s}
            for n, c in configs.items() for s in SEEDS}


def grid(window: str, aspect="stretch", size=384, family="mip_grid") -> str:
    return f"{family}/{window}_{aspect}_{size}"


def stage1() -> dict[str, dict]:
    c = seeded({f"2D-1_{w}": {**MIP, "data_path_key": grid(w)} for w in cfg.MIP_GRID_WINDOWS})
    c.update({
        "3D-0_sphere_r3d18k": {**R3D, **SPHERE},
        "3D-0_sphere_mednet_r18": {**R3D, **SPHERE, **MEDNET, "model": "R3D18"},
        "3D-0_sphere_mednet_r50": {**R3D, **SPHERE, **MEDNET, "model": "R3D50",
                                   "BATCH_SIZE": 2, "GRAD_ACCUM_STEPS": 8},
        "3D-0_sphere_swin_ssl": {**R3D, **SPHERE, **SWIN},
        "3D-1_r3d18k_iso256_mm": {**R3D, "USE_MULTIMODAL": True},
        "3D-2_r3d18k_iso256": {**R3D},
        "3D-3_mednet_r18_iso256": {**R3D, **MEDNET, "model": "R3D18"},
        "3D-4_mednet_r50_iso256": {**R3D, **MEDNET, "model": "R3D50",
                                   "BATCH_SIZE": 2, "GRAD_ACCUM_STEPS": 8},
        "3D-5_r3d18k_vessel90z256": {**R3D, "data_path_key": "vessel90z_256"},
        "3D-6_r3d18k_vessel_crop": {**R3D, "data_path_key": "vessel_crop_256"},
        "3D-7_swin_ssl_c90z256": {**R3D, **SWIN, "data_path_key": "c90z_256"},
        "3D-8_swin_ssl_c90z384": {**R3D, **SWIN, **SINGLE, "data_path_key": "c90z_384"},
    })
    return c


def mean_auc(name: str) -> float:
    path = cfg.EXPERIMENTS_DIR / name / "summary.json"
    if not path.exists():
        sys.exit(f"ERROR: {name} has no summary.json; finish it (or delete its config) first")
    return json.loads(path.read_text())["mean_auc"]


def ranked(prefixes: tuple[str, ...]) -> list[tuple[str, dict]]:
    """Existing final configs whose name starts with a prefix, best first by
    mean val AUC averaged over seed repeats (_s<seed>). Returns base names."""
    seeds = defaultdict(list)
    for p in OUT.glob("*.json"):
        if p.stem.startswith(prefixes):
            seeds[re.sub(r"_s\d+$", "", p.stem)].append(p.stem)
    score = {b: sum(map(mean_auc, ns)) / len(ns) for b, ns in seeds.items()}
    order = sorted(score, key=score.get, reverse=True)
    for b in order:
        print(f"  {score[b]:.4f}  {b}  ({len(seeds[b])} seeds)")
    return [(b, json.loads((OUT / f"{b}.json").read_text())[0]) for b in order]


def window_of(c: dict) -> str:
    return c["data_path_key"].split("/")[1].rsplit("_", 2)[0]


def stage2() -> dict[str, dict]:
    print("2D-1 ranking:")
    top = [window_of(c) for _, c in ranked(("2D-1_",))[:2]]
    c = {f"2D-1b_n4_{top[0]}": {**MIP, "data_path_key": grid(top[0], family="mip_grid_n4")}}
    if (Path(cfg.DATASET_PATHS[grid(top[0], family="mip_grid_n4mask")])).exists():
        c[f"2D-1b_n4mask_{top[0]}"] = {**MIP, "data_path_key": grid(top[0], family="mip_grid_n4mask")}
    for w in top:
        c[f"2D-2_axial_{w}"] = {**MIP, "data_path_key": grid(w), "model": "RESNET50_2D",
                                "INPUT_CHANNELS": [0]}
        c[f"2D-2_latefusion_{w}"] = {**MIP, "data_path_key": grid(w), "model": "RESNET50_2D_LATEFUSION",
                                     "BATCH_SIZE": 16, "GRAD_ACCUM_STEPS": 2}  # 3 ResNet50s: bs32 OOMs on 12 GB
        c[f"2D-2_pad_{w}"] = {**MIP, "data_path_key": grid(w, aspect="pad")}
    return seeded(c)


def stage3() -> dict[str, dict]:
    print("2D-1/1b/2 ranking:")
    name, best = ranked(("2D-1_", "2D-1b_", "2D-2_"))[0]
    print(f"best: {name}")
    fam, variant = best["data_path_key"].split("/")
    w, aspect, _ = variant.rsplit("_", 2)
    c = {f"2D-3_{s}": {**best, "data_path_key": grid(w, aspect, s, fam),
                       "BATCH_SIZE": best["BATCH_SIZE"] * 384 ** 2 // s ** 2} for s in (512, 768)}
    three_ch = {"INPUT_CHANNELS": None}  # 2D-4 backbones take all 3 views even if axial-only won
    c.update({
        "2D-4_swin_t": {**best, **three_ch, "model": "SWIN_T_3CH", "LEARNING_RATE": 3e-5, "WARMUP_EPOCHS": 5},
        "2D-4_mil_tiles": {**best, **three_ch, "model": "MIL_TILES_2D"},
        "2D-4_convnext_t": {**best, **three_ch, "model": "CONVNEXT_T_3CH"},
        "2D-5_multimodal": {**best, "USE_MULTIMODAL": True},
        # Lausanne prefix "laus_" -> train-only; Lausanne-only has its own test split
        "2D-6_mixed": {**best, "data_path_key": grid(w, aspect, 384, "mip_grid_mixed"),
                       "EXTERNAL_DATA_PREFIX": "laus_"},
        "2D-6_lausanne_only": {**best, "data_path_key": grid(w, aspect, 384, "mip_grid_lausanne"),
                               "TEST_IDS_FILE": None},
    })
    if fam != "mip_grid":  # mixed/Lausanne exist for raw MIPs only
        print(f"NOTE: best uses {fam}; 2D-6 falls back to raw MIPs of the same window")
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    args = p.parse_args()
    configs = {1: stage1, 2: stage2, 3: stage3}[args.stage]()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, c in configs.items():
        c = {"name": name, **{k: v for k, v in c.items() if k != "name"}}
        missing = [k for k in c if k not in cfg.DEFAULT_CONFIG]
        assert not missing, f"{name}: unknown keys {missing}"
        assert c["data_path_key"] in cfg.DATASET_PATHS, f"{name}: {c['data_path_key']}"
        (OUT / f"{name}.json").write_text(json.dumps([c], indent=4) + "\n")
    print(f"wrote {len(configs)} configs to {OUT}")


if __name__ == "__main__":
    main()
