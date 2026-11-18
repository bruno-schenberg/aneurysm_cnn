"""
Model registry for the aneurysm CNN training engine.

Adding a new model = one entry in _REGISTRY (dlorchestra.registry.Registry).
The registry stores the constructor and the attribute path to the final
classifier layer (needed for multimodal fusion). Everything else is derived.
"""

import inspect
import logging
from operator import attrgetter
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as image_models
import torchvision.models.video as video_models
from monai.networks.nets import DenseNet, resnet10, resnet18, resnet50

from dlorchestra.registry import Registry

logger = logging.getLogger(__name__)


def _adapt_first_conv(conv: nn.Module, in_channels: int) -> nn.Module:
    """Adapt a pretrained first conv layer (typically 3-channel RGB) to
    in_channels, by averaging its weights across the input-channel dim and
    repeating that average to reach in_channels. Preserves out_channels,
    kernel_size, stride, padding, and bias-presence from conv.

    Works for both nn.Conv2d and nn.Conv3d — conv's own class is reused.
    """
    conv_cls = type(conv)
    new_conv = conv_cls(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    avg_weight = conv.weight.data.mean(dim=1, keepdim=True)  # (out, 1, *kernel_dims)
    repeat_spec = [1, in_channels] + [1] * (avg_weight.dim() - 2)
    new_conv.weight.data = avg_weight.repeat(*repeat_spec)
    return new_conv


# ---------------------------------------------------------------------------
# Model constructors (private — called via the registry)
# ---------------------------------------------------------------------------

def _build_r3d10(num_classes: int) -> nn.Module:
    """MONAI ResNet-10 with MedicalNet pretraining."""
    try:
        model = resnet10(
            pretrained=True, spatial_dims=3, n_input_channels=1,
            feed_forward=False, shortcut_type="B", bias_downsample=False,
        )
        model.fc = nn.Linear(512, num_classes)
        logger.info("R3D10: loaded MedicalNet pretrained weights")
    except Exception as e:
        # No silent scratch fallback: a "MedicalNet" run that quietly trained from
        # scratch would be mislabelled in the thesis (MONAI needs huggingface_hub).
        raise RuntimeError(f"R3D10: MedicalNet weights unavailable: {e}") from e
    return model


def _build_r3d18(num_classes: int) -> nn.Module:
    """MONAI ResNet-18 with MedicalNet pretraining.

    MedicalNet weights require shortcut_type="A" + bias_downsample=True.
    Type A uses zero-padding identity shortcuts (no extra Conv3d+BN),
    which is both faster and lower-memory than type B projections.
    """
    try:
        model = resnet18(
            pretrained=True, spatial_dims=3, n_input_channels=1,
            feed_forward=False, shortcut_type="A", bias_downsample=True,
        )
        model.fc = nn.Linear(512, num_classes)
        logger.info("R3D18: loaded MedicalNet pretrained weights")
    except Exception as e:
        # No silent scratch fallback: a "MedicalNet" run that quietly trained from
        # scratch would be mislabelled in the thesis (MONAI needs huggingface_hub).
        raise RuntimeError(f"R3D18: MedicalNet weights unavailable: {e}") from e
    return model


def _build_r3d18_kinetics(num_classes: int) -> nn.Module:
    """Torchvision R3D-18 (Kinetics-400), adapted to 1 input channel.

    RGB conv1 weights are averaged along the channel dim to produce
    equivalent single-channel filters, preserving learned spatial structure.
    """
    model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
    model.stem[0] = _adapt_first_conv(model.stem[0], in_channels=1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    logger.info("R3D18_KINETICS: loaded Kinetics-400 pretrained weights, adapted to 1ch")
    return model


def _build_r3d50(num_classes: int) -> nn.Module:
    """MONAI ResNet-50 with MedicalNet pretraining."""
    try:
        model = resnet50(
            pretrained=True, spatial_dims=3, n_input_channels=1,
            feed_forward=False, shortcut_type="B", bias_downsample=False,
        )
        model.fc = nn.Linear(2048, num_classes)
        logger.info("R3D50: loaded MedicalNet pretrained weights")
    except Exception as e:
        # No silent scratch fallback: a "MedicalNet" run that quietly trained from
        # scratch would be mislabelled in the thesis (MONAI needs huggingface_hub).
        raise RuntimeError(f"R3D50: MedicalNet weights unavailable: {e}") from e
    return model


def _build_densenet121(num_classes: int) -> nn.Module:
    """MONAI DenseNet-121, trained from scratch."""
    return DenseNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
    )


# ---------------------------------------------------------------------------
# 2D models (for MIP classification)
# ---------------------------------------------------------------------------

def _build_resnet50_2d(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision ResNet-50 (ImageNet), adapted to 1 input channel.

    RGB conv1 weights are averaged along the channel dim.
    """
    model = image_models.resnet50(weights=image_models.ResNet50_Weights.DEFAULT)
    model.conv1 = _adapt_first_conv(model.conv1, in_channels=1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("RESNET50_2D: loaded ImageNet pretrained weights, adapted to 1ch, dropout=%.2f", dropout)
    return model


def _build_resnet50_2d_3ch(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision ResNet-50 (ImageNet), 3 input channels (early fusion MIP).

    Uses default 3-channel conv1 — ImageNet weights used directly.
    """
    model = image_models.resnet50(weights=image_models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("RESNET50_2D_3CH: loaded ImageNet pretrained weights, 3ch (early fusion), dropout=%.2f", dropout)
    return model


class LateFusionResNet50(nn.Module):
    """3x ResNet-50 backbones (one per MIP view) + fusion classifier.

    Input: (B, 3, H, W) where channels are [axial, coronal, sagittal] MIPs.
    Each channel is processed independently through a 1-channel ResNet-50,
    features are concatenated, and a shared classifier produces the output.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbones = nn.ModuleList([
            self._make_backbone() for _ in range(3)
        ])
        feat_dim = 2048  # ResNet-50 feature dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        logger.info(
            "LateFusionResNet50: 3x ResNet-50 backbones (1ch each), "
            "feat_dim=%d*3, classifier 512→%d", feat_dim, num_classes,
        )

    @staticmethod
    def _make_backbone() -> nn.Module:
        model = image_models.resnet50(weights=image_models.ResNet50_Weights.DEFAULT)
        model.conv1 = _adapt_first_conv(model.conv1, in_channels=1)
        model.fc = nn.Identity()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i, backbone in enumerate(self.backbones):
            view = x[:, i : i + 1, :, :]  # (B, 1, H, W)
            features.append(backbone(view))
        return self.classifier(torch.cat(features, dim=1))


def _build_resnet50_2d_latefusion(num_classes: int) -> nn.Module:
    return LateFusionResNet50(num_classes=num_classes)


def _build_resnet50_2d_nch(num_classes: int, in_channels: int = 3, dropout: float = 0.5) -> nn.Module:
    """Torchvision ResNet-50 (ImageNet), adapted to N input channels.

    For thin-slab MIP (5ch), multi-angle MIP (6ch), etc.
    RGB conv1 weights are averaged to 1ch, then replicated to N channels.
    """
    model = image_models.resnet50(weights=image_models.ResNet50_Weights.DEFAULT)
    model.conv1 = _adapt_first_conv(model.conv1, in_channels=in_channels)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("RESNET50_2D_NCH: ImageNet weights, adapted to %dch, dropout=%.2f", in_channels, dropout)
    return model


def _build_efficientnet_b4_3ch(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision EfficientNet-B4 (ImageNet), 3 input channels.

    Natively handles ~380x380 inputs. Better parameter efficiency than ResNet.
    """
    model = image_models.efficientnet_b4(weights=image_models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("EFFICIENTNET_B4_3CH: ImageNet weights, dropout=%.2f", dropout)
    return model


def _build_convnext_t_3ch(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision ConvNeXt-Tiny (ImageNet), 3 input channels.

    Modern CNN competitive with ViTs. Handles arbitrary spatial sizes.
    """
    model = image_models.convnext_tiny(weights=image_models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],  # LayerNorm2d
        model.classifier[1],  # Flatten
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("CONVNEXT_T_3CH: ImageNet weights, dropout=%.2f", dropout)
    return model


def _build_vit_b16_3ch(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision ViT-B/16 (ImageNet), 3 input channels.

    Requires 224x224 input (use RESIZE_TO in config). Patch-based attention
    may focus on small structures better than CNNs.
    """
    model = image_models.vit_b_16(weights=image_models.ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("VIT_B16_3CH: ImageNet weights, dropout=%.2f", dropout)
    return model


def _build_swin_t_3ch(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision Swin-T (ImageNet), 3 input channels.

    Windowed attention with a CNN-like hierarchy, so it takes the native MIP
    sizes (384/512/768, padded to the 7x7 window internally) without the
    224 resize ViT-B/16 needs. Pair with a low LR and WARMUP_EPOCHS.
    """
    model = image_models.swin_t(weights=image_models.Swin_T_Weights.DEFAULT)
    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.head.in_features, num_classes),
    )
    logger.info("SWIN_T_3CH: ImageNet weights, dropout=%.2f", dropout)
    return model


class TileMIL2D(nn.Module):
    """Gated-attention MIL over non-overlapping tiles of a 3-channel MIP.

    The MIP is cut into patch_size² tiles (padded to a multiple), each tile
    goes through a shared ImageNet ResNet-18, and gated attention (Ilse et
    al. 2018) pools the tile features. A 3–5 mm aneurysm fills only a few
    pixels of the whole MIP; here it only has to win the attention within
    the exam, instead of surviving a global average pool. The attention over
    tiles of the last forward pass is kept in ``last_attention`` (B, nH, nW)
    as a heatmap for figures (analysis/predict.py --attention-maps).
    """

    def __init__(self, num_classes: int = 2, patch_size: int = 64, dropout: float = 0.5):
        super().__init__()
        self.patch_size = patch_size
        backbone = image_models.resnet18(weights=image_models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        attn_dim = 128
        self.attn_V = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))
        self.last_attention: torch.Tensor | None = None
        logger.info("MIL_TILES_2D: ResNet-18 ImageNet tiles of %d px, dropout=%.2f", patch_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size
        x = F.pad(x, (0, -W % P, 0, -H % P))
        tiles = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, nH, nW, P, P)
        nH, nW = tiles.shape[2], tiles.shape[3]
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, P, P)

        h = self.backbone(tiles).view(B, nH * nW, -1)
        attn = torch.softmax(self.attn_w(self.attn_V(h) * self.attn_U(h)), dim=1)  # (B, N, 1)
        self.last_attention = attn.detach().view(B, nH, nW)
        return self.classifier((attn * h).sum(dim=1))


def _build_mil_tiles_2d(num_classes: int, patch_size: int = 64, dropout: float = 0.5) -> nn.Module:
    return TileMIL2D(num_classes=num_classes, patch_size=patch_size, dropout=dropout)


# ---------------------------------------------------------------------------
# 2D Slice MIL (slice-level attention aggregation for 3D volumes)
# ---------------------------------------------------------------------------

class SliceMIL2D(nn.Module):
    """ResNet-18 per-slice CNN + gated attention MIL for volume classification.

    Processes a 3D volume as a bag of 2D axial slices:
      1. Each slice -> ResNet-18 -> feature vector
      2. Gated attention weights each slice's contribution
      3. Weighted sum -> classifier head

    Unlike ConvLSTM (which mean-pools LSTM output), attention learns to
    focus on the few slices containing the aneurysm.

    Input: (B, 1, D, H, W) -- standard 3D volume
    Output: (B, num_classes) -- classification logits
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone = image_models.resnet18(weights=image_models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels=1)
        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

        # Gated attention (Ilse et al. 2018)
        attn_dim = 256
        self.attn_V = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        logger.info(
            "SliceMIL2D: ResNet18 (feat_dim=%d) + gated attention (attn_dim=%d)",
            feat_dim, attn_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        chunk_size = 16
        all_features = []
        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            chunk_slices = x[:, :, start:end].permute(0, 2, 1, 3, 4).reshape(
                B * (end - start), C, H, W
            )
            feat = self.backbone(chunk_slices)  # (B*chunk, feat_dim)
            all_features.append(feat.view(B, end - start, self.feat_dim))

        h = torch.cat(all_features, dim=1)  # (B, D, feat_dim)

        # Gated attention
        a_V = self.attn_V(h)  # (B, D, attn_dim)
        a_U = self.attn_U(h)  # (B, D, attn_dim)
        attn_logits = self.attn_w(a_V * a_U)  # (B, D, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, D, 1)

        # Weighted aggregation (attention-pooled)
        bag_repr = (attn_weights * h).sum(dim=1)  # (B, feat_dim)

        return self.classifier(bag_repr)


def _build_mil_slice_2d(num_classes: int) -> nn.Module:
    return SliceMIL2D(num_classes=num_classes)


def _build_resnet50_2d_2ch(num_classes: int, dropout: float = 0.5) -> nn.Module:
    """Torchvision ResNet-50 (ImageNet), adapted to 2 input channels.

    For Frangi 2-channel MIP: [intensity_MIP, frangi_MIP].
    RGB conv1 weights are averaged to 1ch, then duplicated to 2ch.
    """
    model = image_models.resnet50(weights=image_models.ResNet50_Weights.DEFAULT)
    model.conv1 = _adapt_first_conv(model.conv1, in_channels=2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("RESNET50_2D_2CH: loaded ImageNet pretrained weights, adapted to 2ch, dropout=%.2f", dropout)
    return model


# ---------------------------------------------------------------------------
# Conv-LSTM (slice-level aggregation for 3D volumes)
# ---------------------------------------------------------------------------

class ConvLSTM(nn.Module):
    """ResNet-18 per-slice CNN + bidirectional LSTM for volume classification.

    Processes a 3D volume as a sequence of 2D axial slices:
      1. Each slice → frozen/trainable ResNet-18 → feature vector
      2. Sequence of features → BiLSTM → temporal aggregation
      3. Mean-pooled hidden states → classifier head

    Input: (B, 1, D, H, W) — standard 3D volume
    Output: (B, num_classes) — classification logits
    """

    def __init__(self, num_classes: int = 2, lstm_hidden: int = 256, lstm_layers: int = 1):
        super().__init__()
        # 2D ResNet-18 backbone (lighter than ResNet-50 for per-slice processing)
        backbone = image_models.resnet18(weights=image_models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels=1)
        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        logger.info(
            "ConvLSTM: ResNet18 (feat_dim=%d) + BiLSTM (hidden=%d, layers=%d) + classifier",
            feat_dim, lstm_hidden, lstm_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Process slices in chunks to avoid OOM (B*D can be 384+)
        chunk_size = 16
        all_features = []
        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            # (B, 1, chunk, H, W) → (B*chunk, 1, H, W)
            chunk_slices = x[:, :, start:end].permute(0, 2, 1, 3, 4).reshape(
                B * (end - start), C, H, W
            )
            feat = self.backbone(chunk_slices)  # (B*chunk, feat_dim)
            all_features.append(feat.view(B, end - start, self.feat_dim))

        features = torch.cat(all_features, dim=1)  # (B, D, feat_dim)

        # LSTM over slice sequence
        lstm_out, _ = self.lstm(features)  # (B, D, hidden*2)

        # Mean pool over sequence
        pooled = lstm_out.mean(dim=1)  # (B, hidden*2)

        return self.classifier(pooled)


def _build_convlstm(num_classes: int) -> nn.Module:
    return ConvLSTM(num_classes=num_classes, lstm_hidden=256, lstm_layers=1)


# ---------------------------------------------------------------------------
# MIL (Multiple Instance Learning)
# ---------------------------------------------------------------------------

def _build_r3d18_kinetics_backbone() -> tuple[nn.Module, int]:
    """Build R3D-18 Kinetics backbone as a feature extractor (no fc).

    Returns (backbone, feature_dim).
    """
    model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
    model.stem[0] = _adapt_first_conv(model.stem[0], in_channels=1)
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL (Ilse et al. 2018).

    Input: full volume (B, 1, D, H, W).
    Internally extracts non-overlapping 3D patches, runs each through a
    shared backbone, aggregates with gated attention, classifies.
    """

    def __init__(self, num_classes: int = 2, patch_size: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.backbone, feat_dim = _build_r3d18_kinetics_backbone()

        # Gated attention (Ilse et al. 2018, eq. 9)
        attn_dim = 256
        self.attn_V = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        logger.info(
            "GatedAttentionMIL: patch_size=%d, feat_dim=%d, attn_dim=%d",
            patch_size, feat_dim, attn_dim,
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping 3D patches from volume.

        Input: (B, 1, D, H, W) → Output: (B, N, 1, P, P, P)
        where N = (D/P) * (H/P) * (W/P).
        """
        B, C, D, H, W = x.shape
        P = self.patch_size
        patches = (
            x.unfold(2, P, P)   # (B, C, nD, H, W, P)
             .unfold(3, P, P)   # (B, C, nD, nH, W, P, P)
             .unfold(4, P, P)   # (B, C, nD, nH, nW, P, P, P)
        )
        nD, nH, nW = patches.shape[2], patches.shape[3], patches.shape[4]
        # reshape to (B, N, C, P, P, P)
        patches = patches.contiguous().view(B, C, nD * nH * nW, P, P, P)
        patches = patches.permute(0, 2, 1, 3, 4, 5)  # (B, N, C, P, P, P)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self._extract_patches(x)  # (B, N, 1, P, P, P)
        N = patches.shape[1]

        # Process patches sequentially through backbone to save memory
        all_features = []
        for i in range(N):
            patch_batch = patches[:, i]  # (B, 1, P, P, P)
            feat = self.backbone(patch_batch)  # (B, feat_dim)
            all_features.append(feat)
        h = torch.stack(all_features, dim=1)  # (B, N, feat_dim)

        # Gated attention
        a_V = self.attn_V(h)  # (B, N, attn_dim)
        a_U = self.attn_U(h)  # (B, N, attn_dim)
        attn_logits = self.attn_w(a_V * a_U)  # (B, N, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, N, 1)

        # Weighted aggregation
        bag_repr = (attn_weights * h).sum(dim=1)  # (B, feat_dim)

        return self.classifier(bag_repr)


def _build_mil_r3d18k(num_classes: int, patch_size: int = 64) -> nn.Module:
    return GatedAttentionMIL(num_classes=num_classes, patch_size=patch_size)


# ---------------------------------------------------------------------------
# Patch classifier (Phase 7A — cascade stage 2)
# ---------------------------------------------------------------------------

class PatchClassifier3D(nn.Module):
    """Small 3D CNN for classifying nnU-Net candidate patches (aneurysm vs FP).

    Deliberately much smaller than the whole-volume models (R3D18 etc.) —
    the cascade's whole point is that classifying a small, already-localized
    patch is an easier task than finding a tiny target in a full volume, so
    it doesn't need a deep/pretrained backbone. Trained from scratch.

    4 conv blocks halve the spatial size each time (64 -> 32 -> 16 -> 8 -> 4),
    so this expects a 64^3 input (matches build_patch_dataset.py's default
    --patch-size); other sizes divisible by 16 also work via the GAP head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            )

        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        logger.info("PatchClassifier3D: 4 conv blocks (1->16->32->64->128) + GAP, trained from scratch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _build_patch_classifier_3d(num_classes: int, dropout: float = 0.3) -> nn.Module:
    return PatchClassifier3D(num_classes=num_classes, dropout=dropout)


# ---------------------------------------------------------------------------
# SE-ResNet3D (Phase 7B)
# ---------------------------------------------------------------------------

class SEBlock3D(nn.Module):
    """Squeeze-and-excitation channel attention for 3D feature maps.

    Global-average-pools each channel, learns a per-channel gate through a
    small bottleneck MLP, and rescales the input by it (Hu et al. 2018).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1, 1)
        return x * scale


class SEResNet3D(nn.Module):
    """R3D-18 (Kinetics pretrained) with SE channel-attention after each stage.

    The SE blocks are inserted between stages rather than inside the
    pretrained residual blocks themselves, so the Kinetics weights load
    unmodified and only the new SE parameters start from scratch.

    Input: (B, 1, D, H, W) -> Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 2, reduction: int = 16):
        super().__init__()
        backbone, feat_dim = _build_r3d18_kinetics_backbone()
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # torchvision r3d_18 stage output channels: 64, 128, 256, 512
        self.se1 = SEBlock3D(64, reduction)
        self.se2 = SEBlock3D(128, reduction)
        self.se3 = SEBlock3D(256, reduction)
        self.se4 = SEBlock3D(512, reduction)

        self.fc = nn.Linear(feat_dim, num_classes)
        logger.info(
            "SEResNet3D: R3D18 Kinetics backbone + SE blocks (reduction=%d) after each stage",
            reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _build_se_r3d18(num_classes: int, reduction: int = 16) -> nn.Module:
    return SEResNet3D(num_classes=num_classes, reduction=reduction)


# ---------------------------------------------------------------------------
# SwinUNETR encoder → classifier (Phase 7B)
# ---------------------------------------------------------------------------

class SwinEncoderClassifier(nn.Module):
    """MONAI SwinUNETR's Swin Transformer encoder + GAP classifier head.

    Same transfer-learning pattern as NNUNetEncoder (Phase 4A), but with an
    attention-based encoder instead of a plain CNN. Only the encoder half of
    SwinUNETR is built — no decoder, since we only need a global feature for
    classification, not a segmentation map.

    feature_size=48 matches MONAI's public self-supervised SwinViT
    checkpoint (Tang et al. 2022, "Self-Supervised Pretraining of Swin
    Transformers for 3D Medical Images") — set SWIN_CHECKPOINT to point at
    that checkpoint (or a fine-tuned one) to transfer-learn from it, same as
    NNUNET_CHECKPOINT for NNUNetEncoder. Without one, trains from scratch.

    Input spatial dims must be divisible by patch_size**5 = 32 (e.g. our
    c90z_384 volumes: 384x384x192, all multiples of 32).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 feature_size: int = 48, checkpoint_path: str | None = None):
        super().__init__()
        self.swin_vit = self._build_encoder(feature_size)
        self.feat_dim = feature_size * 16  # bottleneck channels (stage 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

    @staticmethod
    def _build_encoder(feature_size: int) -> nn.Module:
        from monai.networks.nets.swin_unetr import SwinTransformer

        return SwinTransformer(
            in_chans=1,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            spatial_dims=3,
        )

    def _load_pretrained(self, checkpoint_path: str):
        """Load SwinViT encoder weights from a MONAI SSL checkpoint.

        MONAI's public SwinViT SSL checkpoints save weights under a
        `state_dict` key. Fine-tuned SwinUNETR checkpoints prefix encoder keys
        with `swinViT.`; the SSL release (model_swinvit.pt, MONAI-extra-test-data
        0.8.1) uses bare encoder keys. Either may carry DDP's `module.`. Keys
        that aren't in the encoder (SSL heads, decoder) are dropped. Raises if
        nothing matches, so a run can't silently train from scratch.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        own_keys = set(self.swin_vit.state_dict())
        encoder_state = {}
        for key, val in state_dict.items():
            clean_key = key.removeprefix("module.").removeprefix("swinViT.")
            # The SSL release predates MONAI renaming the MLP layers fc1/fc2 → linear1/linear2
            clean_key = clean_key.replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2.")
            if clean_key in own_keys:
                encoder_state[clean_key] = val
        if not encoder_state:
            raise ValueError(f"SWIN_ENC: no encoder keys matched in {checkpoint_path}")

        missing, unexpected = self.swin_vit.load_state_dict(encoder_state, strict=False)
        loaded = len(encoder_state) - len(unexpected)
        logger.info(
            "SWIN_ENC: loaded %d pretrained encoder params from %s "
            "(missing=%d, unexpected=%d)",
            loaded, checkpoint_path, len(missing), len(unexpected),
        )
        if missing:
            raise ValueError(f"SWIN_ENC: {len(missing)} encoder keys missing from {checkpoint_path}: {missing[:5]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.swin_vit(x, True)
        bottleneck = hidden_states[-1]  # (B, feat_dim, D', H', W')
        pooled = self.pool(bottleneck).flatten(1)  # (B, feat_dim)
        return self.classifier(pooled)


def _resolve_swin_checkpoint() -> str | None:
    """Resolve SwinViT SSL checkpoint path from env var, if set."""
    import os
    ckpt_path = os.environ.get("SWIN_CHECKPOINT")
    if not ckpt_path:
        logger.info("SWIN_ENC: no SWIN_CHECKPOINT set, training from scratch")
        return None
    if not os.path.exists(ckpt_path):
        # Set but missing = a broken setup, not a request to train from scratch.
        raise FileNotFoundError(f"SWIN_ENC: SWIN_CHECKPOINT={ckpt_path} does not exist")
    logger.info("SWIN_ENC: using checkpoint %s", ckpt_path)
    return ckpt_path


def _build_swin_enc(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return SwinEncoderClassifier(
        num_classes=num_classes, dropout=dropout,
        checkpoint_path=_resolve_swin_checkpoint(),
    )


# ---------------------------------------------------------------------------
# 3D ViT (Phase 7B)
# ---------------------------------------------------------------------------

class ViT3DClassifier(nn.Module):
    """MONAI ViT-B/32 (3D) — plain patch-attention transformer, trained from
    scratch (no locality bias, unlike SWIN_ENC's windowed attention).

    32-voxel patches (~15mm cubes at c90z_384's 0.469mm spacing) keep the
    token count (864 + 1 CLS, for a 384x384x192 volume) tractable for full
    self-attention — a 16-voxel patch would give ~7000 tokens, quadratic in
    the attention matrix. MONAI's ViT.forward returns (logits, hidden_states);
    this wraps it down to just logits to match every other model's
    forward(x) -> logits contract.
    """

    def __init__(self, num_classes: int = 2, img_size: tuple[int, int, int] = (384, 384, 192)):
        super().__init__()
        from monai.networks.nets import ViT

        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=(32, 32, 32),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=True,
            num_classes=num_classes,
            post_activation="none",  # plain Linear head (no Tanh squashing before cross-entropy)
            spatial_dims=3,
        )
        n_tokens = (img_size[0] // 32) * (img_size[1] // 32) * (img_size[2] // 32)
        logger.info(
            "ViT3DClassifier: ViT-B/32 (3D), img_size=%s, patch_size=32^3 (%d tokens+CLS), trained from scratch",
            img_size, n_tokens,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.vit(x)
        return logits


def _build_vit3d_b32(num_classes: int) -> nn.Module:
    return ViT3DClassifier(num_classes=num_classes)


# ---------------------------------------------------------------------------
# nnU-Net encoder → classifier (Phase 4A)
# ---------------------------------------------------------------------------

class NNUNetEncoder(nn.Module):
    """nnU-Net PlainConvUNet encoder with GAP + classifier head.

    Loads the encoder half of a pretrained nnU-Net segmentation model
    (Zenodo 13386859, Dataset610, 3d_fullres) and repurposes it for
    binary classification.

    Architecture (from checkpoint inspection):
      Stage 0: 1 → 32    (3x3x3, pool 1x1x1)
      Stage 1: 32 → 64   (3x3x3, pool 2x2x2)
      Stage 2: 64 → 128  (3x3x3, pool 2x2x2)
      Stage 3: 128 → 256 (3x3x3, pool 2x2x2)
      Stage 4: 256 → 320 (3x3x3, pool 1x2x2)
      Stage 5: 320 → 320 (3x3x3, bottleneck)

    Final feature dim: 320.
    Total encoder params: ~56M (pretrained on 498 TOF-MRA with voxel masks).
    """

    ENCODER_FEAT_DIM = 320

    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 checkpoint_path: str | None = None):
        super().__init__()

        self.encoder = self._build_encoder()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_FEAT_DIM, num_classes),
        )

        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

    def _build_encoder(self) -> nn.Module:
        """Build the PlainConvUNet encoder from dynamic_network_architectures."""
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        # Instantiate full UNet with same config as the Zenodo model,
        # then extract just the encoder.
        unet = PlainConvUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * 6,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
            n_conv_per_stage=[2, 2, 2, 2, 2, 2],
            num_classes=2,
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin_first=False,
        )
        return unet.encoder

    def _load_pretrained(self, checkpoint_path: str):
        """Load encoder weights from nnU-Net checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("network_weights", ckpt)

        # Extract only encoder.* keys, strip the prefix
        encoder_state = {}
        for key, val in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = val

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        loaded = len(encoder_state) - len(unexpected)
        logger.info(
            "NNUNET_ENC: loaded %d pretrained encoder params from %s "
            "(missing=%d, unexpected=%d)",
            loaded, checkpoint_path, len(missing), len(unexpected),
        )
        if missing:
            logger.warning("Missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:5])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder.forward returns a list of feature maps (one per stage)
        # for skip connections. We only need the last (bottleneck) stage.
        skips = self.encoder(x)
        bottleneck = skips[-1]  # (B, 320, D', H', W')
        pooled = self.pool(bottleneck).flatten(1)  # (B, 320)
        return self.classifier(pooled)


class NNUNetEncoderMultiScale(NNUNetEncoder):
    """nnU-Net encoder with multi-scale feature aggregation.

    Instead of only using the bottleneck (stage 5), pools and concatenates
    features from stages 2-5 (128 + 256 + 320 + 320 = 1024 dims).
    Earlier stages have higher spatial resolution and more localized features.
    """

    STAGE_CHANNELS = [32, 64, 128, 256, 320, 320]  # stages 0-5
    USED_STAGES = [2, 3, 4, 5]  # skip stages 0,1 (too early / too large)
    MULTISCALE_DIM = 128 + 256 + 320 + 320  # sum of STAGE_CHANNELS at USED_STAGES

    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 checkpoint_path: str | None = None):
        # super().__init__() builds encoder/pool and loads the checkpoint
        # (which only touches self.encoder's weights); its bottleneck-sized
        # classifier is immediately replaced below with one sized for the
        # concatenated multi-scale features.
        super().__init__(num_classes=num_classes, dropout=dropout, checkpoint_path=checkpoint_path)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.MULTISCALE_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)  # list of 6 feature maps
        pooled = []
        for s in self.USED_STAGES:
            pooled.append(self.pool(skips[s]).flatten(1))
        features = torch.cat(pooled, dim=1)  # (B, 1024)
        return self.classifier(features)


class NNUNetEncoderAttn(NNUNetEncoder):
    """nnU-Net encoder with learned spatial attention pooling.

    Replaces GAP with a 1x1x1 conv that produces per-voxel attention weights.
    The bottleneck features are then weighted-summed, allowing the model to
    focus on specific spatial locations (e.g., where aneurysms are).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 checkpoint_path: str | None = None):
        # super().__init__()'s classifier is already ENCODER_FEAT_DIM ->
        # num_classes — exactly what this subclass needs too, so only the
        # attention conv is added on top.
        super().__init__(num_classes=num_classes, dropout=dropout, checkpoint_path=checkpoint_path)
        self.attn_conv = nn.Conv3d(self.ENCODER_FEAT_DIM, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        bottleneck = skips[-1]  # (B, 320, D', H', W')
        attn_logits = self.attn_conv(bottleneck)  # (B, 1, D', H', W')
        attn_weights = torch.softmax(
            attn_logits.flatten(2), dim=2,
        ).unsqueeze(1)  # (B, 1, 1, D'*H'*W')
        feat_flat = bottleneck.flatten(2)  # (B, 320, D'*H'*W')
        pooled = (feat_flat * attn_weights.squeeze(1)).sum(dim=2)  # (B, 320)
        return self.classifier(pooled)


class NNUNetEncoderMultiScaleAttn(NNUNetEncoder):
    """nnU-Net encoder with multi-scale features + attention pooling.

    Combines both strategies: pools stages 2-5 with per-stage learned
    spatial attention, then concatenates for classification.
    """

    STAGE_CHANNELS = [32, 64, 128, 256, 320, 320]
    USED_STAGES = [2, 3, 4, 5]
    MULTISCALE_DIM = 128 + 256 + 320 + 320  # sum of STAGE_CHANNELS at USED_STAGES

    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 checkpoint_path: str | None = None):
        super().__init__(num_classes=num_classes, dropout=dropout, checkpoint_path=checkpoint_path)
        self.attn_convs = nn.ModuleList([
            nn.Conv3d(self.STAGE_CHANNELS[s], 1, kernel_size=1)
            for s in self.USED_STAGES
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.MULTISCALE_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)  # list of 6 feature maps
        pooled = []
        for attn_conv, s in zip(self.attn_convs, self.USED_STAGES):
            feat = skips[s]  # (B, C_s, D', H', W')
            attn_logits = attn_conv(feat)  # (B, 1, D', H', W')
            attn_weights = torch.softmax(attn_logits.flatten(2), dim=2)  # (B, 1, N)
            feat_flat = feat.flatten(2)  # (B, C_s, N)
            pooled.append((feat_flat * attn_weights).sum(dim=2))  # (B, C_s)
        features = torch.cat(pooled, dim=1)  # (B, 1024)
        return self.classifier(features)


def _resolve_nnunet_checkpoint() -> str | None:
    """Resolve nnU-Net checkpoint path from env var or a repo-relative default.

    Default assumes the usual checkout layout (segmentation/ sibling of
    training_engine/), not a specific machine's home directory.
    """
    import os
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    default_ckpt = str(
        repo_root / "segmentation" / "models" / "nnUNet_results"
        / "Dataset610_AP+DD+ADAM" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0" / "checkpoint_final.pth"
    )
    ckpt_path = os.environ.get("NNUNET_CHECKPOINT", default_ckpt)
    if not os.path.exists(ckpt_path):
        logger.warning("NNUNET_ENC: checkpoint not found at %s, training from scratch", ckpt_path)
        return None
    logger.info("NNUNET_ENC: using checkpoint %s", ckpt_path)
    return ckpt_path


def _build_nnunet_enc(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return NNUNetEncoder(
        num_classes=num_classes, dropout=dropout,
        checkpoint_path=_resolve_nnunet_checkpoint(),
    )


def _build_nnunet_enc_multiscale(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return NNUNetEncoderMultiScale(
        num_classes=num_classes, dropout=dropout,
        checkpoint_path=_resolve_nnunet_checkpoint(),
    )


def _build_nnunet_enc_attn(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return NNUNetEncoderAttn(
        num_classes=num_classes, dropout=dropout,
        checkpoint_path=_resolve_nnunet_checkpoint(),
    )


def _build_nnunet_enc_multiscale_attn(num_classes: int, dropout: float = 0.5) -> nn.Module:
    return NNUNetEncoderMultiScaleAttn(
        num_classes=num_classes, dropout=dropout,
        checkpoint_path=_resolve_nnunet_checkpoint(),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Built on dlorchestra's generic Registry — the pattern (name -> builder +
# metadata) is orchestration infrastructure; every entry below is an
# aneurysm-specific architecture, which is why it stays here.

_REGISTRY: Registry[nn.Module] = Registry()

def _reg(name: str, build: Callable[..., nn.Module], classifier_attr: str) -> None:
    _REGISTRY.register(name, build, classifier_attr=classifier_attr)

_reg("R3D10",           _build_r3d10,           "fc")
_reg("R3D18",           _build_r3d18,           "fc")
_reg("R3D18_KINETICS",  _build_r3d18_kinetics,  "fc")
_reg("R3D50",           _build_r3d50,           "fc")
_reg("DENSENET121",     _build_densenet121,     "class_layers.out")
_reg("MIL_R3D18K",      _build_mil_r3d18k,      "classifier")
_reg("PATCH_CLASSIFIER_3D", _build_patch_classifier_3d, "fc")  # Phase 7A cascade stage 2
# 2D models (MIP classification)
_reg("RESNET50_2D",           _build_resnet50_2d,           "fc")  # fc is Sequential(Dropout, Linear)
_reg("RESNET50_2D_2CH",       _build_resnet50_2d_2ch,       "fc")  # fc is Sequential(Dropout, Linear)
_reg("RESNET50_2D_3CH",       _build_resnet50_2d_3ch,       "fc")  # fc is Sequential(Dropout, Linear)
_reg("RESNET50_2D_NCH",       _build_resnet50_2d_nch,       "fc")  # N-channel (thin-slab, multi-angle)
_reg("RESNET50_2D_LATEFUSION", _build_resnet50_2d_latefusion, "classifier")
# Phase 6 backbones
_reg("EFFICIENTNET_B4_3CH",   _build_efficientnet_b4_3ch,   "classifier")
_reg("CONVNEXT_T_3CH",        _build_convnext_t_3ch,        "classifier")
_reg("VIT_B16_3CH",           _build_vit_b16_3ch,           "heads")
# Final campaign: attention on MIPs
_reg("SWIN_T_3CH",            _build_swin_t_3ch,            "head")
_reg("MIL_TILES_2D",          _build_mil_tiles_2d,          "classifier")
# Slice-level aggregation (3D volume → 2D slices)
_reg("CONVLSTM",              _build_convlstm,              "classifier")
_reg("MIL_SLICE_2D",          _build_mil_slice_2d,          "classifier")
# nnU-Net encoder transfer learning (Phase 4A/4A2)
_reg("NNUNET_ENC",            _build_nnunet_enc,            "classifier")
_reg("NNUNET_ENC_MULTISCALE", _build_nnunet_enc_multiscale, "classifier")
_reg("NNUNET_ENC_ATTN",       _build_nnunet_enc_attn,       "classifier")
_reg("NNUNET_ENC_MULTISCALE_ATTN", _build_nnunet_enc_multiscale_attn, "classifier")
# 3D attention models (Phase 7B)
_reg("SE_R3D18",              _build_se_r3d18,              "fc")
_reg("SWIN_ENC",              _build_swin_enc,              "classifier")
_reg("VIT3D_B32",             _build_vit3d_b32,             "vit.classification_head")

SUPPORTED_MODELS: list[str] = _REGISTRY.names


# ---------------------------------------------------------------------------
# Multimodal late-fusion wrapper
# ---------------------------------------------------------------------------

def _replace_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent_path, _, attr_name = dotted.rpartition(".")
    setattr(attrgetter(parent_path)(root) if parent_path else root, attr_name, new)


def _strip_classifier(model: nn.Module, classifier_attr: str) -> int:
    """Replace the final Linear with Identity, return its in_features.

    The registry's classifier_attr may point at the Linear itself (3D models)
    or at a head that contains it — 2D builders use Sequential(Dropout, Linear),
    ConvNeXt keeps LayerNorm2d + Flatten in front, ViT nests one level deeper.
    Only the last Linear is replaced, so whatever precedes it (pooling,
    flatten, norm) still runs.
    """
    head = attrgetter(classifier_attr)(model)
    if isinstance(head, nn.Linear):
        _replace_submodule(model, classifier_attr, nn.Identity())
        return head.in_features
    linears = [(name, m) for name, m in head.named_modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError(f"No nn.Linear found under '{classifier_attr}'")
    name, last = linears[-1]
    _replace_submodule(head, name, nn.Identity())
    return last.in_features


class MultiModalWrapper(nn.Module):
    """Late fusion: CNN image features + tabular MLP → joint classifier.

    Tabular branch is intentionally shallow (2 layers, 16-dim embedding)
    to avoid overfitting on only 2 features (age, gender).
    """

    def __init__(
        self,
        image_model: nn.Module,
        image_feature_dim: int,
        tabular_dim: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.image_model = image_model
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(image_feature_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_model(image)
        tab_feat = self.tabular_branch(tabular)
        return self.fusion_head(torch.cat([img_feat, tab_feat], dim=1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_head_param_ids(model: nn.Module, model_name: str) -> set[int]:
    """Return id() of parameters belonging to the classifier head.

    For MultiModalWrapper: tabular_branch + fusion_head.
    For GatedAttentionMIL: attention layers + classifier.
    For plain models: the layer at classifier_attr (e.g. model.fc).
    """
    if isinstance(model, MultiModalWrapper):
        head_params = set()
        for p in model.tabular_branch.parameters():
            head_params.add(id(p))
        for p in model.fusion_head.parameters():
            head_params.add(id(p))
        return head_params

    if isinstance(model, GatedAttentionMIL):
        head_params = set()
        for module in (model.attn_V, model.attn_U, model.attn_w, model.classifier):
            for p in module.parameters():
                head_params.add(id(p))
        return head_params

    if isinstance(model, LateFusionResNet50):
        return {id(p) for p in model.classifier.parameters()}

    if isinstance(model, TileMIL2D):
        return {id(p) for m in (model.attn_V, model.attn_U, model.attn_w, model.classifier)
                for p in m.parameters()}

    if isinstance(model, SliceMIL2D):
        head_params = set()
        for module in (model.attn_V, model.attn_U, model.attn_w, model.classifier):
            for p in module.parameters():
                head_params.add(id(p))
        return head_params

    if isinstance(model, ConvLSTM):
        head_params = set()
        for module in (model.lstm, model.classifier):
            for p in module.parameters():
                head_params.add(id(p))
        return head_params

    if isinstance(model, NNUNetEncoderMultiScaleAttn):
        head_params = {id(p) for p in model.classifier.parameters()}
        for p in model.attn_convs.parameters():
            head_params.add(id(p))
        return head_params

    if isinstance(model, NNUNetEncoderAttn):
        head_params = {id(p) for p in model.classifier.parameters()}
        for p in model.attn_conv.parameters():
            head_params.add(id(p))
        return head_params

    if isinstance(model, NNUNetEncoder):
        return {id(p) for p in model.classifier.parameters()}

    if model_name.upper() not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")
    entry = _REGISTRY.get(model_name)

    classifier = attrgetter(entry.metadata["classifier_attr"])(model)
    return {id(p) for p in classifier.parameters()}


def get_model(
    model_name: str,
    num_classes: int = 2,
    use_multimodal: bool = False,
    tabular_dim: int = 2,
    patch_size: int = 64,
    dropout: float = 0.5,
    in_channels: int = 3,
) -> nn.Module:
    """Construct a model by name.

    3D models: input (B, 1, D, H, W) → logits (B, num_classes). 2D/MIP
    models (RESNET50_2D*, EFFICIENTNET_B4_3CH, CONVNEXT_T_3CH, VIT_B16_3CH)
    instead take (B, C, H, W), with C set by in_channels (1/2/3/N depending
    on the variant — see each builder's docstring).
    When use_multimodal=True, the model is wrapped for late fusion and
    forward() expects (image, tabular) instead.
    """
    key = model_name.upper()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}"
        )
    entry = _REGISTRY.get(key)

    # Pass through only the kwargs this builder actually declares, instead
    # of a hand-maintained per-architecture dispatch table — that table
    # silently drifts out of sync (wrong defaults, not a loud failure) if a
    # new registry entry needs one of these kwargs but isn't added to it.
    candidate_kwargs = {"dropout": dropout, "in_channels": in_channels, "patch_size": patch_size}
    accepted = set(inspect.signature(entry.build).parameters)
    build_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    model = entry.build(num_classes=num_classes, **build_kwargs)

    if use_multimodal:
        feat_dim = _strip_classifier(model, entry.metadata["classifier_attr"])
        model = MultiModalWrapper(model, feat_dim, tabular_dim, num_classes)

    return model
