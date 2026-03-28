"""
Tests for train_models.py — experiment config validation and merging.

All tests run without GPU, filesystem access to real datasets, or a running
experiment. Tests exercise prepare_experiment_configs directly with synthetic
experiment dicts and verify that it raises ValueError for invalid inputs and
returns correctly merged configs for valid inputs.
"""

import pytest

from train_models import DATASET_PATHS, DEFAULT_CONFIG, prepare_experiment_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_experiment(**overrides) -> dict:
    """Return a minimal valid experiment dict, with optional field overrides."""
    base = {
        "name": "test_experiment",
        "model": "R3D18",
        "balancing": "none",
        "data_path_key": "A",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    """prepare_experiment_configs must raise ValueError for any missing required key."""

    def test_missing_model_raises(self):
        """Omitting 'model' must raise ValueError naming the missing key."""
        exp = {"name": "x", "balancing": "none", "data_path_key": "A"}
        with pytest.raises(ValueError, match="model"):
            prepare_experiment_configs([exp])

    def test_missing_name_raises(self):
        """Omitting 'name' must raise ValueError naming the missing key."""
        exp = {"model": "R3D18", "balancing": "none", "data_path_key": "A"}
        with pytest.raises(ValueError, match="name"):
            prepare_experiment_configs([exp])

    def test_missing_balancing_raises(self):
        """Omitting 'balancing' must raise ValueError naming the missing key."""
        exp = {"name": "x", "model": "R3D18", "data_path_key": "A"}
        with pytest.raises(ValueError, match="balancing"):
            prepare_experiment_configs([exp])

    def test_missing_data_path_key_raises(self):
        """Omitting 'data_path_key' must raise ValueError naming the missing key."""
        exp = {"name": "x", "model": "R3D18", "balancing": "none"}
        with pytest.raises(ValueError, match="data_path_key"):
            prepare_experiment_configs([exp])


# ---------------------------------------------------------------------------
# Invalid data_path_key
# ---------------------------------------------------------------------------

class TestInvalidDataPathKey:
    """prepare_experiment_configs must raise ValueError for unknown data_path_key values."""

    def test_unknown_key_raises(self):
        """An unrecognised data_path_key must raise ValueError."""
        exp = make_valid_experiment(data_path_key="Z")
        with pytest.raises(ValueError):
            prepare_experiment_configs([exp])

    def test_error_message_names_invalid_key(self):
        """The ValueError message must include the invalid key."""
        exp = make_valid_experiment(data_path_key="Z")
        with pytest.raises(ValueError, match="Z"):
            prepare_experiment_configs([exp])

    def test_error_message_lists_all_valid_keys(self):
        """
        The ValueError message must list all valid dataset keys (A–E), not
        just a subset.  US4 AS#2 requires 'listing the valid keys'.
        """
        exp = make_valid_experiment(data_path_key="X")
        with pytest.raises(ValueError) as exc_info:
            prepare_experiment_configs([exp])
        message = str(exc_info.value)
        for key in ["A", "B", "C", "D", "E"]:
            assert key in message, (
                f"Valid key '{key}' missing from ValueError message. Got: {message}"
            )


# ---------------------------------------------------------------------------
# Valid configuration merging
# ---------------------------------------------------------------------------

class TestValidConfigMerging:
    """prepare_experiment_configs must merge valid experiments correctly."""

    def test_valid_config_returns_list(self):
        """A valid experiment must return a list with one element."""
        result = prepare_experiment_configs([make_valid_experiment()])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_valid_config_contains_all_default_keys(self):
        """Merged config must contain every key from DEFAULT_CONFIG."""
        result = prepare_experiment_configs([make_valid_experiment()])
        config = result[0]
        for key in DEFAULT_CONFIG:
            assert key in config, f"Key '{key}' missing from merged config"

    def test_data_path_is_resolved(self):
        """data_path must be set to the value from DATASET_PATHS[data_path_key]."""
        result = prepare_experiment_configs([make_valid_experiment(data_path_key="A")])
        assert result[0]["data_path"] == DATASET_PATHS["A"]

    def test_experiment_overrides_default(self):
        """Values in the experiment dict must override DEFAULT_CONFIG values."""
        result = prepare_experiment_configs([make_valid_experiment(EPOCHS=10)])
        assert result[0]["EPOCHS"] == 10

    def test_unoverridden_defaults_are_preserved(self):
        """DEFAULT_CONFIG values not overridden must be preserved in merged config."""
        result = prepare_experiment_configs([make_valid_experiment()])
        assert result[0]["EPOCHS"] == DEFAULT_CONFIG["EPOCHS"]
        assert result[0]["RANDOM_SEED"] == DEFAULT_CONFIG["RANDOM_SEED"]

    def test_multiple_valid_experiments(self):
        """Multiple valid experiments must all be processed without error."""
        exps = [
            make_valid_experiment(name="exp_a", data_path_key="A"),
            make_valid_experiment(name="exp_b", data_path_key="B"),
        ]
        result = prepare_experiment_configs(exps)
        assert len(result) == 2
        assert result[0]["name"] == "exp_a"
        assert result[1]["name"] == "exp_b"


# ---------------------------------------------------------------------------
# Dataset path key E (previously missing)
# ---------------------------------------------------------------------------

class TestDatasetPathKeyE:
    """Dataset key E must be recognised after the refactor that added it."""

    def test_key_e_resolves(self):
        """data_path_key='E' must not raise and must resolve to a non-empty path."""
        result = prepare_experiment_configs([make_valid_experiment(data_path_key="E")])
        assert result[0]["data_path"] == DATASET_PATHS["E"]
        assert result[0]["data_path"]  # non-empty string

    def test_all_five_keys_resolve(self):
        """All five dataset keys A–E must resolve without error."""
        for key in ["A", "B", "C", "D", "E"]:
            result = prepare_experiment_configs([make_valid_experiment(data_path_key=key)])
            assert result[0]["data_path"] == DATASET_PATHS[key]
