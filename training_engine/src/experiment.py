"""
Aneurysm classification experiment — wires dlorchestra.TorchExperiment's
generic k-fold fold loop to this project's model registry, dataloaders, and
dataset paths.

All the actual orchestration (logging setup, seeding, k-fold split, AMP
training, metrics/plots/artifacts, completion marker) lives in dlorchestra;
this module only supplies the aneurysm-specific hooks.
"""

import json
import logging
import os
import re
import warnings
from collections import Counter
from pathlib import Path

from dlorchestra.experiment import TorchExperiment

from config import DATASET_CSV, DATASET_PATHS, DATASET_RESOLUTIONS, DEVICE, EXPERIMENTS_DIR
from src.data import (
    _exam_id,
    _load_scanner_lookup,
    build_dataloaders,
    get_data_list,
)
from src.models import get_head_param_ids, get_model
from src.split import exam_key, grouped_split, load_exclusions, patient_of

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


class AneurysmExperiment(TorchExperiment):
    def get_data_list(self, config: dict) -> list[dict]:
        data_path = DATASET_PATHS[config["data_path_key"]]
        data = get_data_list(data_path, use_multimodal=config["USE_MULTIMODAL"])
        logger.info("Loaded %d samples from %s", len(data), data_path)
        if config.get("EXCLUSIONS_FILE"):
            excluded, _ = load_exclusions(config["EXCLUSIONS_FILE"])
            kept = [r for r in data if exam_key(r["path"]) not in excluded]
            logger.info("Excluded %d exams listed in %s", len(data) - len(kept), config["EXCLUSIONS_FILE"])
            data = kept
        return data

    def split_data(self, all_data, config: dict, seed: int, group_fn):
        """With EXCLUSIONS_FILE: frozen test list + patient-grouped folds (src/split.py)."""
        if not config.get("EXCLUSIONS_FILE"):
            return super().split_data(all_data, config, seed, group_fn)
        _, patients = load_exclusions(config["EXCLUSIONS_FILE"])
        keys = [exam_key(r["path"]) for r in all_data]
        ids_file = config.get("TEST_IDS_FILE")
        test_ids = set(json.loads((REPO_ROOT / ids_file).read_text())) if ids_file else None
        folds = grouped_split(
            keys, [r["label"] for r in all_data], [patient_of(k, patients) for k in keys],
            test_ids=test_ids, test_size=config["TEST_SPLIT_RATIO"], seed=seed,
            n_splits=config["N_SPLITS"] if config["USE_KFOLD"] else round(1 / config["VAL_SPLIT_RATIO"]),
            use_kfold=config["USE_KFOLD"],
            strata=[group_fn(r) for r in all_data] if group_fn else None,
        )
        pick = lambda idx: [all_data[i] for i in idx]  # noqa: E731
        return ((k, pick(tr), pick(va), pick(te)) for k, tr, va, te in folds)

    def build_model(self, config: dict, num_classes: int):
        return get_model(
            config["model"],
            num_classes=num_classes,
            use_multimodal=config["USE_MULTIMODAL"],
            patch_size=config.get("MIL_PATCH_SIZE", 64),
            dropout=config.get("DROPOUT", 0.5),
            in_channels=config.get("IN_CHANNELS", 3),
        )

    def build_dataloaders(self, train_files, val_files, test_files, config: dict):
        return build_dataloaders(train_files, val_files, test_files, config)

    def get_head_param_ids(self, model, config: dict):
        return get_head_param_ids(model, config["model"])

    def get_stratify_group_fn(self, config: dict):
        if not config.get("STRATIFY_BY_SCANNER"):
            return None
        if not DATASET_CSV.exists():
            logger.debug("Scanner stratification skipped — DATASET_CSV not found: %s", DATASET_CSV)
            return None
        scanner_lookup = _load_scanner_lookup(DATASET_CSV)
        return lambda rec: scanner_lookup.get(_exam_id(rec), "unknown")

    def check_split(self, fold_idx, train_files, val_files, test_files, config: dict) -> None:
        """Abort before training if the test split differs from TEST_IDS_FILE.

        IDs are matched as BPxxxx[+series letter], so the "ours_" prefix of
        mixed datasets doesn't matter.
        """
        ids_file = config.get("TEST_IDS_FILE")
        if not ids_file or fold_idx > 0:  # the test set is identical across folds
            return
        expected = set(json.loads((REPO_ROOT / ids_file).read_text()))
        actual = {re.search(r"BP\d{4}[A-Z]*", os.path.basename(r["path"])).group() for r in test_files}
        if actual != expected:
            raise ValueError(
                f"Test split differs from {ids_file}: {len(actual - expected)} unexpected, "
                f"{len(expected - actual)} missing (check STRATIFY_BY_SCANNER, SPLIT_SEED, "
                f"TEST_SPLIT_RATIO and the dataset's exam list)"
            )
        logger.info("Test split matches %s (%d exams)", ids_file, len(actual))

    def split_external_data(self, all_data: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
        """Split off records whose filename starts with EXTERNAL_DATA_PREFIX —
        kept in every fold's training set but never in val/test."""
        ext_prefix = config.get("EXTERNAL_DATA_PREFIX")
        if not ext_prefix:
            return all_data, []
        own_data, external_data = [], []
        for rec in all_data:
            fname = os.path.basename(rec["path"])
            if fname.startswith(ext_prefix):
                external_data.append(rec)
            else:
                own_data.append(rec)
        logger.info(
            "External data split: %d own + %d external (prefix='%s')",
            len(own_data), len(external_data), ext_prefix,
        )
        return own_data, external_data

    def log_extra_fold_info(self, train_files, val_files, test_files, config: dict) -> None:
        """Log scanner group counts per split for debugging stratification."""
        if not DATASET_CSV.exists():
            return
        scanner_lookup = _load_scanner_lookup(DATASET_CSV)
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if not files:
                continue
            scanners = [scanner_lookup.get(_exam_id(r), "unknown") for r in files]
            counts = Counter(scanners)
            logger.debug("  %s scanner distribution: %s", split_name, dict(counts.most_common()))

    def validate_config(self, config: dict) -> None:
        """Fail fast on missing or inconsistent config (run-time checks — disk paths etc.).

        Structural checks (required keys, enum values, typos) already ran at
        config-load time via config.validate_config(); this only checks what
        needs the filesystem.
        """
        for key in ("name", "model", "data_path_key"):
            if config.get(key) is None:
                raise ValueError(f"Experiment config missing required key: '{key}'")

        dpk = config["data_path_key"]
        if dpk not in DATASET_PATHS:
            raise ValueError(f"Unknown data_path_key '{dpk}'. Valid: {list(DATASET_PATHS.keys())}")

        data_path = DATASET_PATHS[dpk]
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

        expected_res = DATASET_RESOLUTIONS.get(dpk)
        if expected_res:
            logger.debug("Dataset '%s' expected resolution: %s", dpk, expected_res)

    def silence_known_warnings(self) -> None:
        super().silence_known_warnings()
        warnings.filterwarnings("ignore", message=".*monai.*")
        logging.getLogger("monai").setLevel(logging.WARNING)


def run_experiment(config: dict) -> None:
    """Run a complete cross-validation experiment end-to-end."""
    AneurysmExperiment(experiments_dir=EXPERIMENTS_DIR, device=DEVICE).run(config)
