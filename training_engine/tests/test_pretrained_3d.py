"""Pretrained 3D weights must actually load.

Both used to fail silently and train from scratch: MedicalNet without
huggingface_hub, and the MONAI SwinViT SSL checkpoint (0 of 126 encoder
params matched). Downloads MedicalNet weights on first run; the Swin test
needs SWIN_CHECKPOINT pointing at model_swinvit.pt
(MONAI-extra-test-data release 0.8.1) and is skipped otherwise.

Run: cd training_engine && python -m pytest tests/test_pretrained_3d.py -q
"""
import os

import pytest
import torch

from src.models import get_model


@pytest.mark.parametrize("name", ["R3D18", "R3D50"])
def test_medicalnet_loads(name):
    model = get_model(name)  # raises if the weights can't be fetched
    assert model(torch.randn(1, 1, 32, 64, 64)).shape == (1, 2)


@pytest.mark.skipif(not os.environ.get("SWIN_CHECKPOINT"), reason="SWIN_CHECKPOINT not set")
def test_swin_ssl_loads_every_encoder_key():
    ckpt = torch.load(os.environ["SWIN_CHECKPOINT"], map_location="cpu", weights_only=False)["state_dict"]
    model = get_model("SWIN_ENC")  # raises on any missing encoder key
    assert torch.equal(model.swin_vit.patch_embed.proj.weight, ckpt["module.patch_embed.proj.weight"])
    assert torch.equal(model.swin_vit.layers1[0].blocks[0].mlp.linear1.weight,
                       ckpt["module.layers1.0.blocks.0.mlp.fc1.weight"])
