"""Smoke test for the final-campaign 2D models.

Downloads ImageNet weights on first run (Swin-T ~110 MB, ResNet-18 ~45 MB).
Checks forward/backward at MIP sizes, head-param freezing, and multimodal
wrapping through get_model.

Run: cd training_engine && python -m pytest tests/test_new_2d_models.py -q
"""
import pytest
import torch

from src.models import get_head_param_ids, get_model


@pytest.mark.parametrize("name", ["SWIN_T_3CH", "MIL_TILES_2D"])
@pytest.mark.parametrize("size", [384, 400])  # 400: not a multiple of the tile/window size
def test_forward_backward(name, size):
    model = get_model(name, num_classes=2, patch_size=64)
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 2)
    out.sum().backward()
    head = get_head_param_ids(model, name)
    assert 0 < len(head) < len(list(model.parameters()))


def test_tile_attention_map():
    model = get_model("MIL_TILES_2D", patch_size=64).eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 384, 384))
    a = model.last_attention
    assert a.shape == (1, 6, 6) and torch.allclose(a.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("name", ["SWIN_T_3CH", "MIL_TILES_2D"])
def test_multimodal(name):
    model = get_model(name, use_multimodal=True)
    assert model(torch.randn(2, 3, 128, 128), torch.randn(2, 2)).shape == (2, 2)
