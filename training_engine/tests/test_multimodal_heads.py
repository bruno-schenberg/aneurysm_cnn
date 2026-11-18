"""Multimodal late fusion must work for every head shape in the registry.

2D builders wrap the final Linear in Sequential(Dropout, Linear) (ConvNeXt
also keeps LayerNorm2d + Flatten, ViT nests it one level deeper), so
_strip_classifier must find the last Linear inside the head, not assume the
head itself is one. Built with weights=None: no downloads.

Run: cd training_engine && python -m pytest tests/test_multimodal_heads.py -q
"""
import pytest
import torch
import torch.nn as nn
import torchvision.models as tvm

from src.models import MultiModalWrapper, _strip_classifier


def _resnet50():
    m = tvm.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 2))
    return m, "fc", 2048


def _convnext():
    m = tvm.convnext_tiny(weights=None)
    m.classifier = nn.Sequential(m.classifier[0], m.classifier[1], nn.Dropout(0.5), nn.Linear(768, 2))
    return m, "classifier", 768


def _vit():
    m = tvm.vit_b_16(weights=None)
    m.heads.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 2))
    return m, "heads", 768


def _r3d():
    m = tvm.video.r3d_18(weights=None)
    m.fc = nn.Linear(512, 2)
    return m, "fc", 512


@pytest.mark.parametrize("build, shape", [(_resnet50, (2, 3, 64, 64)), (_convnext, (2, 3, 64, 64)),
                                          (_vit, (2, 3, 224, 224)), (_r3d, (2, 3, 16, 32, 32))])
def test_strip_then_fuse(build, shape):
    model, attr, feat = build()
    assert _strip_classifier(model, attr) == feat
    fused = MultiModalWrapper(model, feat, tabular_dim=2, num_classes=2)
    out = fused(torch.randn(*shape), torch.randn(shape[0], 2))
    assert out.shape == (shape[0], 2)
