"""
Entry point for running training experiments defined in experiments.json.

Reads a JSON file of experiment definitions, merges each entry with
``DEFAULT_CONFIG``, validates required fields, resolves dataset paths,
and hands each fully-formed config to ``orchestrator.run_experiment``.

Usage (from repository root):
    python training_engine/train_models.py
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch

# Redirect MONAI model cache to a user-writable directory (avoids /root/.cache permission issues)
os.environ.setdefault("MONAI_HOME", os.path.expanduser("~/.cache/monai"))

from src.data_preprocess import VALID_RESOLUTIONS
from src.orchestrator import run_experiment


# ── Default Configuration ─────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Fields that must be supplied in experiments.json (None = required) ---
    "name": None,           # Unique experiment identifier; used in output directory names
    "model": None,          # Architecture key; must be one of SUPPORTED_MODELS in models.py
    "balancing": None,      # Class imbalance strategy: 'weighted_cost_function', 'oversampling', or 'none'
    "data_path_key": None,  # Dataset variant key: 'A', 'B', 'C', 'D', or 'E' (see DATASET_PATHS)
    "data_path": None,      # Derived automatically from data_path_key — do not set in JSON

    # --- Tabular metadata (age/gender) fusion — optional late-fusion feature ---
    "USE_TABULAR": False,   # True = fuse age/gender with CNN features (late fusion); False = image-only
    "TABULAR_CSV": None,    # Path to CSV with columns: case_id, age, gender (required when USE_TABULAR=True)

    # --- Training parameters — override any of these in experiments.json as needed ---
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",  # Auto-detected; override with 'cpu' to force CPU
    "RANDOM_SEED": 42,          # Fixed seed for dataset splits, weight init, and shuffle — ensures reproducibility
    "LEARNING_RATE": 0.0001,    # Adam optimiser learning rate; typical range 1e-5 to 1e-3
    "BATCH_SIZE": 4,            # Training mini-batch size; reduce if GPU OOM
    "EPOCHS": 2,                # Number of training epochs per fold
    "N_SPLITS": 5,              # Number of folds for stratified k-fold cross-validation (ignored when USE_KFOLD=False)
    "USE_KFOLD": True,          # True = stratified k-fold CV; False = single 70/30 train/val split
    "VAL_SPLIT_RATIO": 0.30,    # Fraction of dev set used for validation when USE_KFOLD=False
    "EARLY_STOPPING_PATIENCE": 0,  # Epochs without val F2 improvement before stopping; 0 = disabled
    "WEIGHT_DECAY": 1e-4,          # AdamW weight decay (L2 regularisation strength)
    "HOLD_OUT_TEST_SET": True,  # True = reserve TEST_SPLIT_RATIO of data for final evaluation, never seen during training
    "TEST_SPLIT_RATIO": 0.20,   # Fraction of total dataset held out for final evaluation (applies when HOLD_OUT_TEST_SET=True)
    "VAL_BATCH_SIZE": 4,        # Validation and evaluation batch size
    "CLASSES": ["0", "1"],      # Class label strings: '0' = healthy, '1' = aneurysm
    "OUTPUT_DIR": "./experiments",  # Root directory where per-experiment output subdirectories are created
    # --- Input resolution ---
    "INPUT_RESOLUTION": "128x128x128",  # Target spatial size for each volume after the Resized transform;
                                         # parsed by parse_input_resolution into a (H, W, D) tuple for the pipeline.
                                         # Valid values: '128x128x128', '256x256x128', '256x256x256'.
    # --- Gradient accumulation ---
    "GRAD_ACCUM_STEPS": 1,  # Number of mini-batches to accumulate gradients before calling optimizer.step().
                             # 1 = standard per-batch update (default). N > 1 simulates a batch size of
                             # N × BATCH_SIZE without the corresponding memory cost.
}

# Maps data_path_key values from experiments.json to filesystem paths.
# Paths are relative to the training_engine/ working directory.
_DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/data/cases-3")
DATASET_PATHS: Dict[str, str] = {
    "A": f"{_DATA_ROOT}/dataset_A_resampled_cropped",  # 1mm isotropic resample → crop to 128³
    "B": f"{_DATA_ROOT}/dataset_B_resampled_shrunk",   # single-step isotropic resample → pad to 128³
    "C": f"{_DATA_ROOT}/dataset_C_cropped",            # Native resolution → crop to 128³
    "D": f"{_DATA_ROOT}/dataset_D_shrunk",             # Native resolution → shrink to 128³
    "SAMPLE": os.path.expanduser("~/sample_dataset"),  # Synthetic dataset for container/pipeline testing
    "SAMPLE_D": "/mnt/data/nifti-sample-dataset-D",    # 12-case variant-D sample for local integration testing
}


# ── Config Preparation ────────────────────────────────────────────────────────

def prepare_experiment_configs(raw_experiments: List[Dict]) -> List[Dict[str, Any]]:
    """
    Validate and merge a list of raw experiment definitions with ``DEFAULT_CONFIG``.

    Each entry in ``raw_experiments`` is merged with a copy of ``DEFAULT_CONFIG``.
    The following required fields must be present and non-``None`` after merging:
    ``name``, ``model``, ``balancing``, ``data_path_key``.

    The ``data_path`` field is derived automatically from ``data_path_key`` using
    ``DATASET_PATHS``; it must not be set directly in experiments.json.

    Args:
        raw_experiments: List of experiment dicts, typically loaded from
            ``experiments.json``. Each dict may override any key in
            ``DEFAULT_CONFIG``.

    Returns:
        List of fully-formed config dicts, one per experiment, each containing
        all keys from ``DEFAULT_CONFIG`` plus the resolved ``data_path``.

    Raises:
        ValueError: If a required field is missing (names the missing key) or if
            ``data_path_key`` is not a recognised dataset key (names the invalid
            key and lists valid options).
    """
    prepared_configs = []

    for exp_json in raw_experiments:
        config = {**DEFAULT_CONFIG, **exp_json}
        exp_name = config.get("name", "Unnamed")

        # Validate that all required fields have been provided
        required_keys = ["name", "model", "balancing", "data_path_key"]
        for key in required_keys:
            if config[key] is None:
                raise ValueError(
                    f"Required parameter '{key}' is missing in experiment '{exp_name}'."
                )

        # Resolve data_path from data_path_key
        data_key = config["data_path_key"]
        if data_key not in DATASET_PATHS:
            raise ValueError(
                f"'data_path_key' '{data_key}' is not valid in experiment '{exp_name}'. "
                f"Valid keys: {list(DATASET_PATHS.keys())}"
            )
        config["data_path"] = DATASET_PATHS[data_key]

        # Validate tabular config
        if config.get("USE_TABULAR"):
            if not config.get("TABULAR_CSV"):
                raise ValueError(
                    f"'TABULAR_CSV' must be set when 'USE_TABULAR' is True "
                    f"(experiment '{exp_name}')."
                )

        # Validate INPUT_RESOLUTION
        input_res = config.get("INPUT_RESOLUTION", "128x128x128")
        if input_res not in VALID_RESOLUTIONS:
            raise ValueError(
                f"'INPUT_RESOLUTION' '{input_res}' is not valid in experiment '{exp_name}'. "
                f"Valid options: {list(VALID_RESOLUTIONS.keys())}"
            )

        # Validate GRAD_ACCUM_STEPS
        grad_accum = config.get("GRAD_ACCUM_STEPS", 1)
        if not isinstance(grad_accum, int) or isinstance(grad_accum, bool) or grad_accum < 1:
            raise ValueError(
                f"'GRAD_ACCUM_STEPS' must be an integer >= 1 in experiment '{exp_name}'. "
                f"Got: {grad_accum!r}"
            )

        prepared_configs.append(config)

    return prepared_configs


# ── Experiment Runner ─────────────────────────────────────────────────────────

def run_all_experiments(prepared_configs: List[Dict[str, Any]]) -> None:
    """
    Run every experiment in ``prepared_configs`` in sequence.

    Calls ``orchestrator.run_experiment`` for each config. Recovers from
    per-experiment errors (missing data paths, unexpected exceptions) and
    continues to the next experiment rather than aborting the entire run.
    Prints a summary banner when all experiments are complete.

    Args:
        prepared_configs: List of fully-formed experiment config dicts as
            returned by ``prepare_experiment_configs``.
    """
    print(f"Found {len(prepared_configs)} experiments to run.")
    if prepared_configs:
        print(f"Using device: {prepared_configs[0]['DEVICE']}")

    for exp_config in prepared_configs:
        try:
            run_experiment(exp_config)
        except FileNotFoundError as e:
            print(f"\nFATAL ERROR: Data path missing for experiment '{exp_config['name']}'. {e}")
        except Exception as e:
            print(f"\nFATAL ERROR running experiment '{exp_config['name']}': {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n\n" + "*" * 80)
    print("EXPERIMENTS FINISHED")
    print("*" * 80)


# ── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        default="experiments.json",
        help="Path to experiments JSON file (default: experiments.json)",
    )
    args = parser.parse_args()

    # Resolve relative paths against the script's directory
    if not os.path.isabs(args.experiments):
        experiments_file = os.path.join(os.path.dirname(__file__), args.experiments)
    else:
        experiments_file = args.experiments
    try:
        with open(experiments_file) as f:
            experiments_to_run = json.load(f)
    except FileNotFoundError:
        print(f"Error: Experiments file not found at '{experiments_file}'")
        raise SystemExit(1)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from '{experiments_file}'. "
            "Please check its format."
        )
        raise SystemExit(1)

    prepared_configs = prepare_experiment_configs(experiments_to_run)
    run_all_experiments(prepared_configs)
