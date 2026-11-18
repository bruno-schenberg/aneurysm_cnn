"""Score a trained experiment's fold checkpoints on any dataset.

Rebuilds the model from the experiment's config.json, loads every
fold_*/checkpoints/model_best.pth, runs the same eval transforms as training,
and writes one row per exam: per-fold probabilities and their mean (the
fold ensemble). Uses: ours → Lausanne and Lausanne → ours cross-scoring
(2D-6), re-scoring the test set, external datasets.

Usage:
    python -m analysis.predict --experiment experiments/NAME --out preds.csv \\
        [--data DIR] [--ids ../dataset/test_ids.json] [--attention-maps DIR] [--device cuda]

--data defaults to the experiment's own dataset. Multimodal experiments
need age/sex for every exam in DATASET_CSV, so they can't score Lausanne.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset

import config as cfg
from src.data import get_data_list, get_transforms
from src.experiment import AneurysmExperiment
from src.models import TileMIL2D


def exam_id(path: str) -> str:
    """BPxxxx[A-Z] for our exams (any prefix), else the file stem (Lausanne)."""
    name = Path(path).name.removesuffix(".gz").removesuffix(".nii").removesuffix(".npy")
    m = re.search(r"BP\d{4}[A-Z]*", name)
    return m.group() if m else name


def load_fold_models(exp_dir: Path, config: dict, device: str) -> list[tuple[str, torch.nn.Module]]:
    builder = AneurysmExperiment(experiments_dir=exp_dir.parent, device=device)
    models = []
    for ckpt in sorted(exp_dir.glob("fold_*/checkpoints/model_best.pth")):
        model = builder.build_model(config, num_classes=len(config["CLASSES"]))
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}  # torch.compile prefix
        model.load_state_dict(state)
        models.append((ckpt.parts[-3], model.to(device).eval()))
    if not models:
        raise FileNotFoundError(f"No fold_*/checkpoints/model_best.pth under {exp_dir}")
    return models


def predict(exp_dir: Path, data_dir: Path | None = None, ids: set[str] | None = None,
            device: str = "cpu", attention_dir: Path | None = None) -> pd.DataFrame:
    config = {**cfg.DEFAULT_CONFIG, **json.loads((exp_dir / "config.json").read_text())}
    data_dir = Path(data_dir or cfg.DATASET_PATHS[config["data_path_key"]])
    try:
        files = get_data_list(data_dir, use_multimodal=config["USE_MULTIMODAL"])
    except KeyError as e:
        raise SystemExit(f"Multimodal experiment, but {e}: these exams have no age/sex in DATASET_CSV") from e
    if ids is not None:
        files = [f for f in files if exam_id(f["path"]) in ids]
        missing = ids - {exam_id(f["path"]) for f in files}
        if missing:
            raise SystemExit(f"{len(missing)} requested exams not in {data_dir}: {sorted(missing)[:5]}")

    use_tabular = config["USE_MULTIMODAL"]
    loader = DataLoader(Dataset(files, get_transforms(config, augment=False, use_tabular=use_tabular)),
                        batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])
    use_amp = config["USE_AMP"] and device.startswith("cuda")
    amp_dtype = torch.float16 if config.get("AMP_DTYPE") == "float16" else torch.bfloat16

    out = pd.DataFrame({"exam_id": [exam_id(f["path"]) for f in files],
                        "image": [f["path"] for f in files],
                        "true_label": [f["label"] for f in files]})
    for fold, model in load_fold_models(exp_dir, config, device):
        probs, maps = [], []
        with torch.no_grad(), torch.autocast(device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            for batch in loader:
                x = batch["image"].to(device)
                logits = model(x, batch["tabular"].to(device)) if use_tabular else model(x)
                probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
                if attention_dir is not None and isinstance(model, TileMIL2D):
                    maps.append(model.last_attention.cpu())
        out[f"prob_{fold}"] = torch.cat(probs).numpy()
        if maps:
            attention_dir.mkdir(parents=True, exist_ok=True)
            for eid, a in zip(out.exam_id, torch.cat(maps).numpy()):
                np.save(attention_dir / f"{eid}_{fold}.npy", a)
    out["prob_mean"] = out.filter(like="prob_fold").mean(axis=1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--ids", type=Path, help="JSON list of exam IDs to keep (e.g. ../dataset/test_ids.json)")
    p.add_argument("--attention-maps", type=Path, help="save MIL_TILES_2D tile attention per exam/fold here")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    ids = set(json.loads(a.ids.read_text())) if a.ids else None
    df = predict(a.experiment, a.data, ids, a.device, a.attention_maps)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} exams × {df.filter(like='prob_fold').shape[1]} folds -> {a.out}")


if __name__ == "__main__":
    main()
