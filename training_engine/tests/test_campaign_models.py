"""Every model the final campaign uses: forward + backward + head freezing.

Small inputs on CPU; downloads pretrained weights on first run (~0.6 GB).
SWIN_ENC uses SWIN_CHECKPOINT if set, else scratch (only shapes are checked).

Run: cd training_engine && python -m pytest tests/test_campaign_models.py -q
"""
import pytest
import torch

from src.models import get_head_param_ids, get_model

CAMPAIGN = [  # (model, input shape)
    ("RESNET50_2D_3CH", (2, 3, 96, 96)),         # 2D-1/1b/3/5/6/7
    ("RESNET50_2D", (2, 1, 96, 96)),             # 2D-2 axial-only
    ("RESNET50_2D_LATEFUSION", (2, 3, 96, 96)),  # 2D-2 late fusion
    ("SWIN_T_3CH", (2, 3, 96, 96)),              # 2D-4
    ("MIL_TILES_2D", (2, 3, 128, 128)),          # 2D-4
    ("CONVNEXT_T_3CH", (2, 3, 96, 96)),          # 2D-4
    ("R3D18_KINETICS", (1, 1, 32, 64, 64)),      # 3D-0/1/2/5/6
    ("R3D18", (1, 1, 32, 64, 64)),               # 3D-0/3 MedicalNet
    ("R3D50", (1, 1, 32, 64, 64)),               # 3D-0/4 MedicalNet
    ("SWIN_ENC", (1, 1, 64, 64, 64)),            # 3D-0/7/8 (dims must be multiples of 32)
]


@pytest.mark.parametrize("name, shape", CAMPAIGN)
@pytest.mark.parametrize("multimodal", [False, True])
def test_model(name, shape, multimodal):
    model = get_model(name, use_multimodal=multimodal, in_channels=shape[1])
    x = torch.randn(*shape)
    out = model(x, torch.randn(shape[0], 2)) if multimodal else model(x)
    assert out.shape == (shape[0], 2)
    out.sum().backward()
    head = get_head_param_ids(model, name)
    assert 0 < len(head) < len(list(model.parameters()))
