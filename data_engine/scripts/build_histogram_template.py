"""
Build a reference intensity template from Skyra 3T volumes.

Saves a 1-D NumPy array of quantile values that can be loaded at training
time to apply histogram matching as a transform (no full dataset duplication).

Usage:
    python data_engine/scripts/build_histogram_template.py
    python data_engine/scripts/build_histogram_template.py --variant iso_128 --n-reference 50
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import config

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
log = logging.getLogger(__name__)

REFERENCE_SCANNER = "skyra"
REFERENCE_FIELD = 3.0
N_BINS = 1024


def _get_reference_exams(n: int, rng: np.random.Generator) -> list[str]:
    df = pd.read_csv(config.DATASET_CSV)
    skyra = df[
        (df["scanner"] == REFERENCE_SCANNER)
        & (df["field_strength_T"] == REFERENCE_FIELD)
        & (df["class"].isin([0, 1]))
    ]
    ids = skyra["exam"].tolist()
    rng.shuffle(ids)
    return ids[:n]


def _find_volume(dataset_dir: Path, exam_id: str) -> Path | None:
    for cls in ("0", "1"):
        p = dataset_dir / cls / f"{exam_id}.nii.gz"
        if p.exists():
            return p
    return None


def build_template(dataset_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    exam_ids = _get_reference_exams(n, rng)
    log.info("Building template from %d Skyra 3T volumes in %s", len(exam_ids), dataset_dir)

    templates: list[np.ndarray] = []
    for eid in exam_ids:
        path = _find_volume(dataset_dir, eid)
        if path is None:
            log.warning("Volume not found for %s — skipping", eid)
            continue
        vol = nib.load(path).get_fdata(dtype=np.float32)
        fg = vol[vol > 0]
        if len(fg) == 0:
            continue
        quantiles = np.linspace(0, 1, N_BINS)
        sorted_vals = np.quantile(fg, quantiles)
        templates.append(sorted_vals)

    if not templates:
        raise RuntimeError(f"No Skyra reference volumes found in {dataset_dir}")

    avg_template = np.mean(templates, axis=0).astype(np.float32)
    log.info("Template built from %d volumes (%d bins)", len(templates), N_BINS)
    return avg_template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build histogram reference template.")
    p.add_argument("--variant", default="iso_128",
                    help="Dataset variant to sample from (default: iso_128)")
    p.add_argument("--n-reference", type=int, default=999,
                    help="Number of Skyra volumes to average (default: 999 = all available)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    p.add_argument("--output", default=None,
                    help="Output .npy path (default: data_engine/dataset/histogram_template.npy)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset_dir = config.DATASETS_ROOT / args.variant

    if not dataset_dir.exists():
        log.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    template = build_template(dataset_dir, args.n_reference, rng)

    out_path = Path(args.output) if args.output else config.DATASET_CSV.parent / "histogram_template.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, template)
    log.info("Saved template to %s (%d bytes)", out_path, out_path.stat().st_size)


if __name__ == "__main__":
    main()
