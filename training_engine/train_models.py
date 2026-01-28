import json
import os
import torch
from src.orchestrator import run_experiment
from typing import List, Dict, Any

# --- 1. Default Configuration ---
DEFAULT_CONFIG = {
    # --- Parameters to be defined in experiments.json ---
    "name": None,
    "model": None,
    "balancing": None,
    "data_path_key": None,
    "data_path": None, # Derived from data_path_key

    # --- Default training parameters (can be overridden in JSON) ---
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RANDOM_SEED": 42,
    "LEARNING_RATE": 0.0001,
    "BATCH_SIZE": 4,
    "EPOCHS": 2,
    "N_SPLITS": 2,
    "QUICK_TEST": True,      # Set to True to run only one fold for quick testing
    "HOLD_OUT_TEST_SET": True, # If True, per-fold evaluation uses the val set, not the test set.
    "TEST_SPLIT_RATIO": 0.20,  # Ratio of the total dataset to be held out for testing
    "VAL_BATCH_SIZE": 4,
    "CLASSES": ['0', '1'],  # 0: healthy, 1: aneurysm
    "OUTPUT_DIR": './experiments'
}

# This dictionary maps keys from the JSON file to actual file paths.
DATASET_PATHS = {
    "A": 'datasets/resample_crop',
    "B": 'datasets/resample_shrink',
    "C": 'datasets/no_resample_crop',
    "D": 'datasets/no_resample_shrink',
    "test": 'test'
}

# --- 2. Configuration Preparation ---
def prepare_experiment_configs(raw_experiments: List[Dict]) -> List[Dict[str, Any]]:
    """
    Creates a list of configuration dictionaries from raw experiment definitions,
    merging them with defaults and resolving data paths.
    """
    prepared_configs = []
    for exp_json in raw_experiments:
        try:
            # Start with a copy of defaults and merge the experiment's JSON config
            config = {**DEFAULT_CONFIG, **exp_json}

            # Validate that required keys from JSON are now filled
            required_keys = ["name", "model", "balancing", "data_path_key"]
            for key in required_keys:
                if config[key] is None:
                    raise ValueError(f"Required parameter '{key}' is missing.")

            # Validate data_path_key and set the final data_path
            data_key = config['data_path_key']
            if data_key in DATASET_PATHS:
                config['data_path'] = DATASET_PATHS[data_key]
            else:
                raise ValueError(f"'data_path_key' '{data_key}' is not a valid dataset key.")

            prepared_configs.append(config)

        except (KeyError, ValueError) as e:
            print(f"Error creating config for experiment '{exp_json.get('name', 'Unnamed')}': {e}")
            exit(1)
    return prepared_configs

# --- 3. Main Orchestrator ---
def run_all_experiments(prepared_configs: List[Dict[str, Any]]):
    """
    Iterates through a list of experiment configurations and runs each experiment.
    """
    print(f"Found {len(prepared_configs)} experiments to run.")
    if prepared_configs:
        print(f"Using device: {prepared_configs[0]['DEVICE']}")

    for exp_config in prepared_configs:
        try:
            run_experiment(exp_config)
        except FileNotFoundError as e:
            print(f"\nFATAL ERROR: Data path missing for experiment {exp_config['name']}. {e}")
        except Exception as e:
            print(f"\nFATAL ERROR running experiment {exp_config['name']}: {e}")

    print("\n\n" + "*"*80)
    print("EXPERIMENTS FINISHED")
    print("*"*80)

# --- 4. Main Execution Block ---
if __name__ == "__main__":

    # --- Load Experiments from JSON ---
    experiments_file = os.path.join(os.path.dirname(__file__), '.', 'experiments.json')
    try:
        with open(experiments_file, 'r') as f:
            experiments_to_run = json.load(f)
    except FileNotFoundError:
        print(f"Error: Experiments file not found at '{experiments_file}'")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{experiments_file}'. Please check its format.")
        exit(1)

    # --- Prepare All Experiment Configurations ---
    prepared_configs = prepare_experiment_configs(experiments_to_run)

    # --- Run All Defined Experiments ---    
    run_all_experiments(prepared_configs)