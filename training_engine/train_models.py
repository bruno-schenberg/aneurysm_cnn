"""
CLI entry point for the training engine.

Usage:
    cd training_engine
    python train_models.py --experiments configs/smoke_test.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dlorchestra.experiment import is_experiment_complete

from config import CONFIGS_DIR, DEFAULT_CONFIG, EXPERIMENTS_DIR, validate_config
from src.experiment import run_experiment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_experiment_configs(json_path: Path) -> list[dict]:
    """Load a flat JSON list of experiment dicts and merge each over DEFAULT_CONFIG.

    Each entry must have at least: name, model, data_path_key.
    All other keys fall back to DEFAULT_CONFIG.
    """
    with open(json_path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of experiment dicts, got {type(raw).__name__}. "
            f"File: {json_path}"
        )

    configs = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} in {json_path} is not a dict")
        config = {**DEFAULT_CONFIG, **entry}
        validate_config(config)
        configs.append(config)

    return configs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_experiments(configs: list[dict]) -> list[str]:
    """Run each experiment, skipping completed ones. Per-experiment exception isolation.

    Returns the names of experiments that failed.
    """
    failed = []
    for config in configs:
        name = config["name"]
        base_name = config.get("_base_name")

        if is_experiment_complete(EXPERIMENTS_DIR, name, base_name):
            logger.info("Skipping '%s' — already complete (summary.json exists)", name)
            continue

        logger.info("=" * 60)
        logger.info("Starting experiment: %s", name)
        logger.info("=" * 60)

        try:
            run_experiment(config)
        except Exception:
            logger.exception("Experiment '%s' failed", name)
            failed.append(name)
    return failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train aneurysm classification models",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        required=True,
        help="Path to a JSON file containing a list of experiment configs",
    )
    args = parser.parse_args()

    # Resolve path — try as-is first, then relative to configs/
    json_path = Path(args.experiments)
    if not json_path.exists():
        json_path = CONFIGS_DIR / args.experiments
    if not json_path.exists():
        print(f"Error: experiment file not found: {args.experiments}", file=sys.stderr)
        sys.exit(1)

    configs = load_experiment_configs(json_path)

    # Append SLURM job ID to experiment names for traceability
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        for config in configs:
            config["_base_name"] = config["name"]
            config["name"] = f"{config['name']}_{slurm_job_id}"
        logger.info("SLURM job %s — appended job ID to experiment names", slurm_job_id)

    logger.info("Loaded %d experiment(s) from %s", len(configs), json_path)

    failed = run_all_experiments(configs)
    if failed:
        logger.error("%d experiment(s) failed: %s", len(failed), ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    # Basic root logger for startup messages (experiment runner reconfigures per-experiment)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
