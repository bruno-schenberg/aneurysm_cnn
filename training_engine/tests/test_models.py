"""
Tests for models.py — construction and forward pass for every registered model.

All tests run on CPU with a synthetic zero-filled input tensor.
No GPU, no internet connection, and no real NIfTI files are required.

MONAI ResNet models (R3D50, UNet3DWithBackbone, MIL_R3D18) attempt to download
MedicalNet pretrained weights on first construction. The implementation falls back
to random initialisation if the download fails, so tests pass in offline environments.
"""

import pytest
import torch

from src.models import SUPPORTED_MODELS, get_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE = (1, 1, 128, 128, 128)
NUM_CLASSES = 2
EXPECTED_OUTPUT_SHAPE = (1, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_input() -> torch.Tensor:
    """Zero-filled tensor with the standard single-channel 128³ input shape."""
    return torch.zeros(*INPUT_SHAPE)


# ---------------------------------------------------------------------------
# Parametrized forward-pass test — covers every entry in SUPPORTED_MODELS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_all_supported_models_forward_pass(model_name: str):
    """
    Every model in SUPPORTED_MODELS must construct without error and produce
    the correct output shape on CPU for a (1, 1, 128, 128, 128) input.
    """
    model = get_model(model_name, num_classes=NUM_CLASSES)
    model.eval()

    x = make_input()
    with torch.no_grad():
        output = model(x)

    assert output.shape == EXPECTED_OUTPUT_SHAPE, (
        f"{model_name}: expected output shape {EXPECTED_OUTPUT_SHAPE}, got {output.shape}"
    )


# ---------------------------------------------------------------------------
# Case-insensitive resolution
# ---------------------------------------------------------------------------

def test_get_model_lowercase_resolves():
    """get_model must accept a fully lowercase model name."""
    model = get_model("r3d18", num_classes=NUM_CLASSES)
    assert model is not None


def test_get_model_uppercase_resolves():
    """get_model must accept a fully uppercase model name."""
    model = get_model("R3D18", num_classes=NUM_CLASSES)
    assert model is not None


def test_get_model_mixed_case_resolves():
    """get_model must accept a mixed-case model name."""
    model = get_model("DenseNet121", num_classes=NUM_CLASSES)
    assert model is not None


# ---------------------------------------------------------------------------
# Unknown model name
# ---------------------------------------------------------------------------

def test_get_model_unknown_name_raises_value_error():
    """get_model must raise ValueError for an unrecognised model name."""
    with pytest.raises(ValueError):
        get_model("unknown_architecture", num_classes=NUM_CLASSES)


def test_get_model_error_message_lists_valid_names():
    """The ValueError message must reference SUPPORTED_MODELS."""
    with pytest.raises(ValueError, match="R3D18"):
        get_model("not_a_model", num_classes=NUM_CLASSES)


# ---------------------------------------------------------------------------
# SUPPORTED_MODELS registry completeness
# ---------------------------------------------------------------------------

def test_supported_models_is_non_empty():
    """SUPPORTED_MODELS must list at least one model."""
    assert len(SUPPORTED_MODELS) > 0


def test_supported_models_contains_spec_required_models():
    """
    SUPPORTED_MODELS must contain every model named explicitly in the spec
    acceptance scenarios (US1 AS#1–5).
    """
    spec_required = {"R3D18", "R3D50", "DenseNet121", "SwinUNETR", "UNet3D"}
    missing = spec_required - set(SUPPORTED_MODELS)
    assert not missing, f"SUPPORTED_MODELS is missing spec-required models: {missing}"


def test_get_model_error_message_lists_all_supported_models():
    """
    The ValueError for an unknown model name must reference every name in
    SUPPORTED_MODELS, not just a subset.  US1 AS#7 requires 'listing all valid
    names from SUPPORTED_MODELS'.
    """
    with pytest.raises(ValueError) as exc_info:
        get_model("not_a_real_model", num_classes=NUM_CLASSES)
    message = str(exc_info.value)
    for name in SUPPORTED_MODELS:
        assert name in message, (
            f"'{name}' missing from ValueError message. Got: {message}"
        )


def test_every_supported_model_name_resolves():
    """
    Every name in SUPPORTED_MODELS must be accepted by get_model without raising.
    This is a belt-and-suspenders check that SUPPORTED_MODELS and the routing
    table in get_model are kept in sync.
    """
    for name in SUPPORTED_MODELS:
        model = get_model(name, num_classes=NUM_CLASSES)
        assert model is not None, f"get_model('{name}') returned None"
