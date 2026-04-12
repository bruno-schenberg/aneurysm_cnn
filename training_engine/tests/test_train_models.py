"""
Tests for train_models.py — experiment config validation and merging.

All tests run without GPU, filesystem access to real datasets, or a running
experiment. Tests exercise prepare_experiment_configs directly with synthetic
experiment dicts and verify that it raises ValueError for invalid inputs and
returns correctly merged configs for valid inputs.
"""

import pytest

from train_models import DATASET_PATHS, DATASET_RESOLUTIONS, DEFAULT_CONFIG, prepare_experiment_configs


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
        for key in ["A", "B", "C", "D", "A256", "B256", "C256", "D256"]:
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
# INPUT_RESOLUTION config validation
# ---------------------------------------------------------------------------

class TestInputResolutionConfig:
    """prepare_experiment_configs must validate and default INPUT_RESOLUTION."""

    def test_absent_input_resolution_defaults_to_128(self):
        """When INPUT_RESOLUTION is absent, merged config must default to '128x128x128'."""
        result = prepare_experiment_configs([make_valid_experiment()])
        assert result[0]["INPUT_RESOLUTION"] == "128x128x128"

    def test_128x128x128_is_valid(self):
        """'128x128x128' must be accepted without error."""
        result = prepare_experiment_configs([make_valid_experiment(INPUT_RESOLUTION="128x128x128")])
        assert result[0]["INPUT_RESOLUTION"] == "128x128x128"

    def test_256x256x128_is_valid(self):
        """'256x256x128' must be accepted when paired with a 256-dataset key."""
        result = prepare_experiment_configs([make_valid_experiment(
            data_path_key="A256", INPUT_RESOLUTION="256x256x128"
        )])
        assert result[0]["INPUT_RESOLUTION"] == "256x256x128"

    def test_256x256x256_is_a_recognised_resolution_string(self):
        """'256x256x256' is a valid resolution string in VALID_RESOLUTIONS (no dataset key yet)."""
        from src.data_preprocess import VALID_RESOLUTIONS
        assert "256x256x256" in VALID_RESOLUTIONS

    def test_invalid_resolution_raises_value_error(self):
        """An unrecognised INPUT_RESOLUTION must raise ValueError before any data loads."""
        with pytest.raises(ValueError):
            prepare_experiment_configs([make_valid_experiment(INPUT_RESOLUTION="512x512x512")])

    def test_error_message_names_invalid_resolution(self):
        """The ValueError message must include the invalid resolution string."""
        with pytest.raises(ValueError, match="bad_res"):
            prepare_experiment_configs([make_valid_experiment(INPUT_RESOLUTION="bad_res")])

    def test_256_resolution_with_128_dataset_raises(self):
        """Using INPUT_RESOLUTION='256x256x128' with a 128³ dataset must raise ValueError."""
        with pytest.raises(ValueError, match="256x256x128"):
            prepare_experiment_configs([make_valid_experiment(
                data_path_key="A", INPUT_RESOLUTION="256x256x128"
            )])

    def test_128_resolution_with_256_dataset_raises(self):
        """Using INPUT_RESOLUTION='128x128x128' with a 256×256×128 dataset must raise ValueError."""
        with pytest.raises(ValueError, match="128x128x128"):
            prepare_experiment_configs([make_valid_experiment(
                data_path_key="A256", INPUT_RESOLUTION="128x128x128"
            )])

    def test_256_resolution_with_256_dataset_passes(self):
        """Using INPUT_RESOLUTION='256x256x128' with a 256×256×128 dataset must not raise."""
        result = prepare_experiment_configs([make_valid_experiment(
            data_path_key="A256", INPUT_RESOLUTION="256x256x128"
        )])
        assert result[0]["INPUT_RESOLUTION"] == "256x256x128"
        assert result[0]["data_path"] == DATASET_PATHS["A256"]

    def test_all_256_keys_require_256_resolution(self):
        """All 256 dataset keys must require INPUT_RESOLUTION='256x256x128'."""
        for key in ["A256", "B256", "C256", "D256"]:
            assert DATASET_RESOLUTIONS[key] == "256x256x128"
            result = prepare_experiment_configs([make_valid_experiment(
                data_path_key=key, INPUT_RESOLUTION="256x256x128"
            )])
            assert result[0]["data_path"] == DATASET_PATHS[key]

    def test_all_128_keys_require_128_resolution(self):
        """All 128³ dataset keys must require INPUT_RESOLUTION='128x128x128'."""
        for key in ["A", "B", "C", "D"]:
            assert DATASET_RESOLUTIONS[key] == "128x128x128"
            result = prepare_experiment_configs([make_valid_experiment(data_path_key=key)])
            assert result[0]["INPUT_RESOLUTION"] == "128x128x128"


# ---------------------------------------------------------------------------
# GRAD_ACCUM_STEPS config validation
# ---------------------------------------------------------------------------

class TestGradAccumConfig:
    """prepare_experiment_configs must validate GRAD_ACCUM_STEPS."""

    def test_absent_grad_accum_steps_defaults_to_1(self):
        """When GRAD_ACCUM_STEPS is absent, merged config must default to 1."""
        result = prepare_experiment_configs([make_valid_experiment()])
        assert result[0]["GRAD_ACCUM_STEPS"] == 1

    def test_valid_grad_accum_steps_4(self):
        """GRAD_ACCUM_STEPS=4 must be accepted without error."""
        result = prepare_experiment_configs([make_valid_experiment(GRAD_ACCUM_STEPS=4)])
        assert result[0]["GRAD_ACCUM_STEPS"] == 4

    def test_zero_raises_value_error(self):
        """GRAD_ACCUM_STEPS=0 must raise ValueError."""
        with pytest.raises(ValueError):
            prepare_experiment_configs([make_valid_experiment(GRAD_ACCUM_STEPS=0)])

    def test_negative_raises_value_error(self):
        """GRAD_ACCUM_STEPS=-1 must raise ValueError."""
        with pytest.raises(ValueError):
            prepare_experiment_configs([make_valid_experiment(GRAD_ACCUM_STEPS=-1)])

    def test_float_raises_value_error(self):
        """GRAD_ACCUM_STEPS=2.5 (a float) must raise ValueError."""
        with pytest.raises(ValueError):
            prepare_experiment_configs([make_valid_experiment(GRAD_ACCUM_STEPS=2.5)])


