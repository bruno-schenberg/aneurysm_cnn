"""
Tests for training.py — F2 checkpointing criterion.

All tests run on CPU with mock DataLoaders. No GPU or real data required.
"""

import pytest
import torch

import src.training as training_module
from src.training import run_training_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_model() -> torch.nn.Module:
    """A minimal 2-class linear model for CPU-only testing."""
    return torch.nn.Linear(4, 2)


def make_mock_loader():
    """A single-batch DataLoader-like list yielding the dict format the loops expect."""
    return [{"image": torch.zeros(2, 4), "label": torch.tensor([0, 1])}]


# ---------------------------------------------------------------------------
# F2 Checkpointing
# ---------------------------------------------------------------------------

class TestF2Checkpointing:
    """run_training_loop must checkpoint the epoch with highest val_f2, not lowest val_loss."""

    def test_checkpoints_f2_optimal_epoch(self, monkeypatch):
        """
        Epoch 2 has higher val_loss but better val_f2 than epoch 1 and 3.
        The saved checkpoint must correspond to epoch 2.
        """
        # (val_loss, val_acc, val_f2) returned by validate_one_epoch per epoch
        val_returns = [
            (0.30, 0.70, 0.40),  # epoch 1 — best loss, mediocre F2
            (0.50, 0.65, 0.85),  # epoch 2 — worse loss, best F2 → must be checkpointed
            (0.20, 0.80, 0.30),  # epoch 3 — best loss of all, worst F2
        ]
        call_state = {"n": 0}

        def mock_validate(model, dataloader, criterion, device, return_details=False):
            result = val_returns[call_state["n"]]
            call_state["n"] += 1
            return result

        monkeypatch.setattr(training_module, "validate_one_epoch", mock_validate)
        monkeypatch.setattr(
            training_module, "train_one_epoch", lambda *a, **kw: (0.5, 0.7)
        )

        model = make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        metrics_history, _, best_checkpoint = run_training_loop(
            model=model,
            train_loader=make_mock_loader(),
            val_loader=make_mock_loader(),
            criterion_cls=torch.nn.CrossEntropyLoss,
            optimizer=optimizer,
            device="cpu",
            num_epochs=3,
        )

        assert best_checkpoint is not None
        assert best_checkpoint["best_epoch"] == 2
        assert best_checkpoint["best_val_f2"] == pytest.approx(0.85)

    def test_metrics_history_contains_val_f2(self, monkeypatch):
        """Each entry in metrics_history includes the val_f2 field."""
        monkeypatch.setattr(
            training_module,
            "validate_one_epoch",
            lambda *a, **kw: (0.4, 0.7, 0.6),
        )
        monkeypatch.setattr(
            training_module, "train_one_epoch", lambda *a, **kw: (0.5, 0.7)
        )

        model = make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        metrics_history, _, _ = run_training_loop(
            model=model,
            train_loader=make_mock_loader(),
            val_loader=make_mock_loader(),
            criterion_cls=torch.nn.CrossEntropyLoss,
            optimizer=optimizer,
            device="cpu",
            num_epochs=2,
        )

        assert len(metrics_history) == 2
        for record in metrics_history:
            assert "val_f2" in record
            assert "epoch" in record
            assert "train_loss" in record
            assert "val_loss" in record

    def test_checkpoint_is_none_when_all_f2_zero(self, monkeypatch):
        """When every epoch produces val_f2 == 0, no checkpoint is saved."""
        monkeypatch.setattr(
            training_module,
            "validate_one_epoch",
            lambda *a, **kw: (0.5, 0.6, 0.0),
        )
        monkeypatch.setattr(
            training_module, "train_one_epoch", lambda *a, **kw: (0.5, 0.6)
        )

        model = make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        _, _, best_checkpoint = run_training_loop(
            model=model,
            train_loader=make_mock_loader(),
            val_loader=make_mock_loader(),
            criterion_cls=torch.nn.CrossEntropyLoss,
            optimizer=optimizer,
            device="cpu",
            num_epochs=2,
        )

        assert best_checkpoint is None

    def test_checkpoint_metadata_reflects_f2_optimal_epoch(self, monkeypatch):
        """
        The checkpoint metadata (best_epoch, best_val_f2) must correspond to the
        epoch with the highest val_f2, and the checkpoint must contain a state_dict key.
        """
        val_returns = [
            (0.4, 0.7, 0.9),  # epoch 1 — best F2
            (0.3, 0.8, 0.1),  # epoch 2 — better loss, worse F2
        ]
        call_state = {"n": 0}

        def mock_validate(model, dataloader, criterion, device, return_details=False):
            result = val_returns[call_state["n"]]
            call_state["n"] += 1
            return result

        monkeypatch.setattr(training_module, "validate_one_epoch", mock_validate)
        monkeypatch.setattr(
            training_module, "train_one_epoch", lambda *a, **kw: (0.5, 0.7)
        )

        model = make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        _, _, best_checkpoint = run_training_loop(
            model=model,
            train_loader=make_mock_loader(),
            val_loader=make_mock_loader(),
            criterion_cls=torch.nn.CrossEntropyLoss,
            optimizer=optimizer,
            device="cpu",
            num_epochs=2,
        )

        assert best_checkpoint is not None
        assert best_checkpoint["best_epoch"] == 1
        assert best_checkpoint["best_val_f2"] == pytest.approx(0.9)
        assert "state_dict" in best_checkpoint

    def test_criterion_cls_must_be_class_not_instance(self):
        """Passing an instantiated criterion raises TypeError."""
        model = make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with pytest.raises(TypeError, match="criterion_cls must be the class type"):
            run_training_loop(
                model=model,
                train_loader=make_mock_loader(),
                val_loader=make_mock_loader(),
                criterion_cls=torch.nn.CrossEntropyLoss(),  # instance, not class
                optimizer=optimizer,
                device="cpu",
                num_epochs=1,
            )
