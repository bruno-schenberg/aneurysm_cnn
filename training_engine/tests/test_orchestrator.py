"""
Tests for orchestrator.py — fold execution flow and eval loader selection.

Uses mock loaders and a stub model. No GPU or real data required.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.orchestrator import run_one_fold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    return {
        "N_SPLITS": 5,
        "EPOCHS": 1,
        "LEARNING_RATE": 0.001,
        "DEVICE": "cpu",
        "CLASSES": ["0", "1"],
        "name": "test_experiment",
    }


@pytest.fixture
def stub_loaders():
    return MagicMock(), MagicMock(), MagicMock()  # train, val, test


@pytest.fixture
def tiny_model():
    """Real linear model so Adam and load_state_dict work without errors."""
    return torch.nn.Linear(4, 2)


@pytest.fixture
def fake_checkpoint(tiny_model):
    return {
        "state_dict": tiny_model.state_dict(),
        "best_epoch": 1,
        "best_val_f2": 0.75,
    }


# ---------------------------------------------------------------------------
# HOLD_OUT_TEST_SET routing
# ---------------------------------------------------------------------------


class TestEvalLoaderSelection:

    def _run_with_mocks(self, config, model, stub_loaders, fake_checkpoint, tmp_path):
        train_loader, val_loader, test_loader = stub_loaders

        with (
            patch("src.orchestrator.run_training_loop") as mock_train,
            patch("src.orchestrator.validate_one_epoch") as mock_val,
            patch("src.orchestrator.save_fold_artifacts"),
            patch(
                "src.orchestrator.calculate_classification_metrics", return_value={}
            ),
        ):
            mock_train.return_value = ([], 0.1, fake_checkpoint)
            mock_val.return_value = (0.5, 0.7, 0.6, [])

            run_one_fold(
                fold=0,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=model,
                config=config,
                weights_tensor=None,
                experiment_output_dir=str(tmp_path),
            )

            return mock_val

    def test_hold_out_true_routes_to_test_loader(
        self, tmp_path, minimal_config, stub_loaders, tiny_model, fake_checkpoint
    ):
        """When HOLD_OUT_TEST_SET=True, final evaluation must use test_loader."""
        minimal_config["HOLD_OUT_TEST_SET"] = True
        _, val_loader, test_loader = stub_loaders

        mock_val = self._run_with_mocks(
            minimal_config, tiny_model, stub_loaders, fake_checkpoint, tmp_path
        )

        final_call_args = mock_val.call_args[0]
        actual_loader = final_call_args[1]
        assert actual_loader is test_loader, (
            "Expected test_loader but got a different loader when HOLD_OUT_TEST_SET=True."
        )

    def test_hold_out_false_routes_to_val_loader(
        self, tmp_path, minimal_config, stub_loaders, tiny_model, fake_checkpoint
    ):
        """When HOLD_OUT_TEST_SET=False, final evaluation must use val_loader."""
        minimal_config["HOLD_OUT_TEST_SET"] = False
        _, val_loader, test_loader = stub_loaders

        mock_val = self._run_with_mocks(
            minimal_config, tiny_model, stub_loaders, fake_checkpoint, tmp_path
        )

        final_call_args = mock_val.call_args[0]
        actual_loader = final_call_args[1]
        assert actual_loader is val_loader, (
            "Expected val_loader but got a different loader when HOLD_OUT_TEST_SET=False."
        )

    def test_checkpoint_metadata_json_written_when_checkpoint_exists(
        self, tmp_path, minimal_config, stub_loaders, tiny_model, fake_checkpoint
    ):
        """best_checkpoint_metadata.json is written when a best checkpoint exists."""
        import json

        minimal_config["HOLD_OUT_TEST_SET"] = True
        self._run_with_mocks(
            minimal_config, tiny_model, stub_loaders, fake_checkpoint, tmp_path
        )

        metadata_path = tmp_path / "fold_0" / "best_checkpoint_metadata.json"
        assert metadata_path.exists(), "best_checkpoint_metadata.json was not created."

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["best_epoch"] == fake_checkpoint["best_epoch"]
        assert abs(metadata["best_val_f2"] - fake_checkpoint["best_val_f2"]) < 1e-6

    def test_no_metadata_json_when_no_checkpoint(
        self, tmp_path, minimal_config, stub_loaders, tiny_model
    ):
        """best_checkpoint_metadata.json is NOT written when no checkpoint was saved."""
        minimal_config["HOLD_OUT_TEST_SET"] = True
        train_loader, val_loader, test_loader = stub_loaders

        with (
            patch("src.orchestrator.run_training_loop") as mock_train,
            patch("src.orchestrator.validate_one_epoch") as mock_val,
            patch("src.orchestrator.save_fold_artifacts"),
            patch(
                "src.orchestrator.calculate_classification_metrics", return_value={}
            ),
        ):
            mock_train.return_value = ([], 0.1, None)  # no checkpoint
            mock_val.return_value = (0.5, 0.7, 0.0, [])

            run_one_fold(
                fold=0,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=tiny_model,
                config=minimal_config,
                weights_tensor=None,
                experiment_output_dir=str(tmp_path),
            )

        metadata_path = tmp_path / "fold_0" / "best_checkpoint_metadata.json"
        assert not metadata_path.exists(), (
            "best_checkpoint_metadata.json should not be written when no checkpoint exists."
        )
