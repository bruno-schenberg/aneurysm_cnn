"""
Model registry for the aneurysm CNN training engine.

Every architecture in SUPPORTED_MODELS accepts a single-channel 128³ input tensor
(B, 1, 128, 128, 128) and returns class logits (B, num_classes).

Pretrained weights are sourced from:
- Kinetics-400 video dataset (torchvision R3D-18)
- MedicalNet / Med3D (MONAI ResNet-18 and ResNet-50, 23 medical imaging datasets)
- BTCV multi-organ segmentation (SwinUNETR, loaded from a local cache path)

All other architectures (DenseNet-121, UNet3D) are trained from scratch.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from monai.networks.nets import DenseNet, SwinUNETR, resnet18, resnet50


# ── Model Registry ─────────────────────────────────────────────────────────────

SUPPORTED_MODELS: list[str] = [
    "R3D18",            # torchvision R3D-18 — Kinetics-400 pretrained, adapted to 1 channel
    "R3D50",            # MONAI ResNet-50   — MedicalNet pretrained (23 medical datasets)
    "DenseNet121",      # MONAI DenseNet-121 — trained from scratch
    "SwinUNETR",        # MONAI SwinUNETR encoder + classification head
    "UNet3D",           # Custom 3D UNet classifier — trained from scratch
    "UNet3DWithBackbone",  # Custom 3D UNet with MONAI ResNet-18 encoder (MedicalNet)
    "MIL_R3D18",        # Attention MIL with MONAI ResNet-18 instance extractor
]


# ── 1. R3D-18 Models ──────────────────────────────────────────────────────────

def get_r3d18_pytorch_model(
    num_classes: int, weights: Optional[str] = "default"
) -> torch.nn.Module:
    """
    Torchvision R3D-18 adapted for single-channel 3D medical image classification.

    Architecture: 18-layer 3D ResNet (R3D) from torchvision's video model zoo.
    Input shape: (B, 1, D, H, W) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: Kinetics-400 video dataset (RGB, 3 channels).
    Single-channel adaptation: the pretrained first-layer weights are averaged
    across the 3 input channels to produce a single-channel kernel, preserving
    the spatial filter structure without discarding learned features.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        weights: ``'default'`` loads Kinetics-400 pretrained weights; any other
            value initialises the model with random weights.
    """
    if weights == "default":
        model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
    else:
        model = video_models.r3d_18(weights=None)

    # Replace the first conv layer: 3 channels → 1 channel.
    # Average pretrained RGB weights along the channel dimension so the learned
    # spatial filters are preserved rather than discarded.
    original_conv1 = model.stem[0]
    new_conv1 = nn.Conv3d(
        in_channels=1,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None,
    )
    new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
    model.stem[0] = new_conv1

    # Replace the classification head with one sized for our task.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_r3d18_monai_model(num_classes: int) -> torch.nn.Module:
    """
    MONAI ResNet-18 for single-channel 3D medical image classification.

    Architecture: 18-layer 3D ResNet from ``monai.networks.nets``.
    Input shape: (B, 1, D, H, W) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: MedicalNet / Med3D — a ResNet pre-trained on 23 medical
    imaging datasets (Tencent MedicalNet project). Weights are downloaded
    automatically from ``TencentMedicalNet/MedicalNet-Resnet18`` on Hugging Face
    Hub when first used. Requires MONAI >= 1.3.1.

    If pretrained weights cannot be downloaded (e.g., offline environment),
    the model falls back to random initialisation so that construction always
    succeeds.

    The original 400-class classification head is replaced with a fresh linear
    layer sized for the target task.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
    """
    try:
        model = resnet18(pretrained=True, spatial_dims=3, n_input_channels=1, num_classes=400)
    except Exception:
        # Fall back to random initialisation if the pretrained download fails
        # (e.g., no internet connection or Hugging Face Hub unavailable).
        model = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=400)

    # Replace the 400-class MedicalNet head with a task-specific head.
    # ResNet-18 encoder produces 512-dimensional features.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ── 2. R3D-50 Model ───────────────────────────────────────────────────────────

def get_r3d50_model(num_classes: int) -> torch.nn.Module:
    """
    MONAI ResNet-50 for single-channel 3D medical image classification.

    Architecture: 50-layer 3D ResNet (bottleneck blocks) from ``monai.networks.nets``.
    Input shape: (B, 1, D, H, W) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: MedicalNet / Med3D — a ResNet pre-trained on 23 medical
    imaging datasets (Tencent MedicalNet project). Weights are downloaded
    automatically from ``TencentMedicalNet/MedicalNet-Resnet50`` on Hugging Face
    Hub when first used. Requires MONAI >= 1.3.1.

    If pretrained weights cannot be downloaded (e.g., offline environment),
    the model falls back to random initialisation so that construction always
    succeeds.

    The original 400-class classification head is replaced with a fresh linear
    layer sized for the target task.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
    """
    try:
        model = resnet50(pretrained=True, spatial_dims=3, n_input_channels=1, num_classes=400)
    except Exception:
        # Fall back to random initialisation if the pretrained download fails.
        model = resnet50(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=400)

    # Replace the 400-class MedicalNet head with a task-specific head.
    # ResNet-50 bottleneck encoder produces 2048-dimensional features.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ── 3. DenseNet-121 Model ─────────────────────────────────────────────────────

def get_densenet121_monai_model(num_classes: int) -> torch.nn.Module:
    """
    MONAI DenseNet-121 for single-channel 3D medical image classification.

    Architecture: 3D DenseNet-121 from ``monai.networks.nets``, configured with
    the canonical DenseNet-121 growth parameters (init_features=64,
    growth_rate=32, block_config=(6, 12, 24, 16)).
    Input shape: (B, 1, D, H, W) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: None — this model is initialised with random weights and
    trained entirely from scratch on the aneurysm dataset.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
    """
    model = DenseNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        # DenseNet-121 architecture parameters
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
    )
    return model


# ── 4. SwinUNETR Model ────────────────────────────────────────────────────────

class SwinUNETRClassifier(nn.Module):
    """
    SwinUNETR encoder repurposed for 3D volumetric classification.

    The SwinUNETR backbone (a hierarchical Swin Transformer) is used purely as a
    feature extractor. Its segmentation decoder is discarded. The deepest encoder
    output — a (B, H, W, D, feature_size*16) tensor — is global-average-pooled
    and fed into a linear classification head.

    Architecture: MONAI SwinUNETR with ``feature_size=48``.
    Input shape: (B, 1, 128, 128, 128) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: SwinUNETR trained on the BTCV dataset for multi-organ
    segmentation (14 classes). Only the Swin Transformer encoder weights
    (``swinViT``) are loaded; the decoder is not used. Weights are read from
    ``/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth``. If the
    file is absent, the model falls back to random initialisation and a warning
    is printed — construction and forward pass still succeed.

    Requires ``einops`` (``pip install einops``) for the Swin Transformer
    window-partitioning operations. Add ``einops`` to your conda environment.

    Note: MONAI >= 1.4 removed the ``img_size`` constructor argument. The
    window-based attention in SwinUNETR is now position-independent and accepts
    inputs of any spatial size without pre-specifying ``img_size``.
    """

    def __init__(
        self,
        img_size: tuple,
        in_channels: int,
        num_classes: int,
        feature_size: int = 48,
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.feature_size = feature_size

        # Instantiate the full SwinUNETR (needed to load pretrained encoder weights).
        # ``out_channels=14`` matches the pretrained BTCV segmentation checkpoint.
        # Note: MONAI >= 1.4 removed the ``img_size`` argument; the window-based
        # attention in SwinUNETR is position-independent and works for any input size.
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=14,
            feature_size=self.feature_size,
            use_checkpoint=True,
        )

        if use_pretrained:
            print("  -> Loading pre-trained weights for SwinUNETR backbone...")
            try:
                # Only the Swin Transformer encoder sub-module (swinViT) is loaded;
                # the decoder weights are discarded since we replace it with GAP + linear.
                model_dict = torch.load(
                    "/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth",
                    weights_only=True,
                )
                state_dict = model_dict["state_dict"]
                self.swin_unetr.swinViT.load_state_dict(state_dict)
                print("  -> SwinUNETR backbone weights loaded successfully.")
            except FileNotFoundError:
                print(
                    "  -> Warning: Pre-trained weights not found. "
                    "The model will be trained from scratch."
                )
            except Exception as e:
                print(
                    f"  -> Error loading pre-trained weights: {e}. "
                    "The model will be trained from scratch."
                )

        # Classification head: global average pooling followed by a linear layer.
        # The deepest SwinTransformer output has feature_size * 16 channels (768 for
        # feature_size=48) and a spatial layout of (H/32, W/32, D/32).
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(self.feature_size * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swin_unetr.swinViT returns a list of 5 multi-scale feature tensors
        # (x0_out through x4_out) in NCDHW layout.
        # x4_out (index -1) is the deepest representation:
        #   shape (B, feature_size*16, D/32, H/32, W/32) e.g. (B, 768, 4, 4, 4)
        hidden_states = self.swin_unetr.swinViT(x)
        encoder_output = hidden_states[-1]  # (B, feature_size*16, ...)

        pooled = self.global_pool(encoder_output)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)


def get_swinunetr_model(num_classes: int) -> torch.nn.Module:
    """
    Constructs a ``SwinUNETRClassifier`` configured for 128³ single-channel inputs.

    See ``SwinUNETRClassifier`` for full architecture and pretrained weight details.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
    """
    return SwinUNETRClassifier(
        img_size=(128, 128, 128),  # kept for API compatibility; not passed to SwinUNETR internally
        in_channels=1,
        num_classes=num_classes,
        use_pretrained=True,
    )


# ── 5. UNet3D Models ──────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """
    Standard double-convolution block used in the UNet encoder and decoder paths.

    Each block applies two sequential 3×3×3 convolutions, each followed by
    BatchNorm and ReLU activation. This is the canonical UNet building block.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels (same for both convolutions).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3DClassifier(nn.Module):
    """
    Custom 3D UNet adapted for volumetric classification.

    The full encoder-decoder path with skip connections is preserved, but the
    final segmentation output is replaced with Global Average Pooling (GAP) and
    a fully connected classification head. This leverages the UNet's multi-scale
    feature hierarchy for classification rather than pixel-wise prediction.

    Architecture: 4 encoder stages (each halving spatial resolution) → bottleneck
    → 4 decoder stages (each doubling spatial resolution via transposed convolution
    + skip concatenation) → GAP → linear head.

    Input shape: (B, 1, 128, 128, 128) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: None — trained from scratch on the aneurysm dataset.

    Args:
        in_channels: Number of input channels (1 for single-channel MRI/CT).
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        base_c: Base channel count. Channel counts at each level are multiples of
            ``base_c`` (e.g., base_c, base_c*2, base_c*4, base_c*8, base_c*16).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_c: int = 32):
        super().__init__()

        # Encoder path — four downsampling stages
        self.enc1 = ConvBlock3D(in_channels, base_c)        # 128³
        self.pool1 = nn.MaxPool3d(2)                         # → 64³
        self.enc2 = ConvBlock3D(base_c, base_c * 2)         # 64³
        self.pool2 = nn.MaxPool3d(2)                         # → 32³
        self.enc3 = ConvBlock3D(base_c * 2, base_c * 4)     # 32³
        self.pool3 = nn.MaxPool3d(2)                         # → 16³
        self.enc4 = ConvBlock3D(base_c * 4, base_c * 8)     # 16³
        self.pool4 = nn.MaxPool3d(2)                         # → 8³

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_c * 8, base_c * 16)  # 8³ feature map

        # Decoder path — four upsampling stages with skip connections
        self.up4 = nn.ConvTranspose3d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_c * 16, base_c * 8)   # cat(up4, enc4) = base_c*16
        self.up3 = nn.ConvTranspose3d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_c * 8, base_c * 4)    # cat(up3, enc3) = base_c*8
        self.up2 = nn.ConvTranspose3d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_c * 4, base_c * 2)    # cat(up2, enc2) = base_c*4
        self.up1 = nn.ConvTranspose3d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_c * 2, base_c)        # cat(up1, enc1) = base_c*2

        # Classification head — GAP collapses the final feature map to a vector
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(e4))

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat((e4, self.up4(bottleneck)), dim=1))
        d3 = self.dec3(torch.cat((e3, self.up3(d4)), dim=1))
        d2 = self.dec2(torch.cat((e2, self.up2(d3)), dim=1))
        d1 = self.dec1(torch.cat((e1, self.up1(d2)), dim=1))

        # Classification head — (B, C, 1, 1, 1) → (B, C) → (B, num_classes)
        pooled = self.global_pool(d1)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)


class UNet3DWithBackbone(nn.Module):
    """
    3D UNet-like architecture with a pretrained MONAI ResNet-18 encoder.

    The encoder is a MONAI ResNet-18 pretrained on MedicalNet (23 medical datasets).
    Its four residual-block stages provide skip-connection features for a standard
    UNet decoder. A Global Average Pooling head on the final decoded feature map
    produces class logits.

    This combines the inductive bias of UNet skip connections with the feature
    quality of a medically pretrained encoder, without requiring segmentation labels.

    Architecture: ResNet-18 encoder (layer1–layer4) → UNet decoder
    (ConvTranspose3d + skip concat) → GAP → linear head.

    Input shape: (B, 1, 128, 128, 128) — single-channel volumetric scan.
    Output shape: (B, num_classes) — raw class logits.

    Pretrained weights: MedicalNet encoder (see ``get_r3d18_monai_model``).
    Decoder weights are initialised randomly.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        base_c: Base channel count for the decoder; does not affect the encoder.
    """

    def __init__(self, num_classes: int = 2, base_c: int = 32):
        super().__init__()

        # Encoder: pretrained MONAI ResNet-18 (MedicalNet weights).
        # ResNet-18 layer output channels: layer1=64, layer2=128, layer3=256, layer4=512.
        self.backbone = get_r3d18_monai_model(num_classes=num_classes)

        # Decoder path — channel sizes match the ResNet-18 layer outputs.
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(512, 256)   # cat(up3=256, layer3=256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(256, 128)   # cat(up2=128, layer2=128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(128, 64)    # cat(up1=64, layer1=64)

        self.up0 = nn.ConvTranspose3d(64, base_c, kernel_size=2, stride=2)
        self.dec0 = ConvBlock3D(base_c + 64, base_c)  # cat(up0=base_c, stem=64)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the backbone layer-by-layer to collect skip-connection features.
        # MONAI ResNet stores its activation as .act (not .relu).
        e0 = self.backbone.act(self.backbone.bn1(self.backbone.conv1(x)))
        e1 = self.backbone.layer1(self.backbone.maxpool(e0))
        e2 = self.backbone.layer2(e1)
        e3 = self.backbone.layer3(e2)
        bottleneck = self.backbone.layer4(e3)

        # Decoder path with skip connections
        d3 = self.dec3(torch.cat((e3, self.up3(bottleneck)), dim=1))
        d2 = self.dec2(torch.cat((e2, self.up2(d3)), dim=1))
        d1 = self.dec1(torch.cat((e1, self.up1(d2)), dim=1))
        d0 = self.dec0(torch.cat((e0, self.up0(d1)), dim=1))

        # Classification head
        pooled = self.global_pool(d0)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)


def get_unet3d_with_backbone_model(num_classes: int) -> torch.nn.Module:
    """
    Constructs a ``UNet3DWithBackbone`` with a MONAI ResNet-18 encoder.

    See ``UNet3DWithBackbone`` for full architecture and pretrained weight details.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
    """
    return UNet3DWithBackbone(num_classes=num_classes, base_c=32)


# ── 6. Multiple Instance Learning (MIL) Model ─────────────────────────────────

class MILClassifier(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) classifier.

    MIL is suited to cases where a single scan can be decomposed into overlapping
    patches (instances). The model aggregates patch-level features into a single
    scan-level (bag-level) representation using learned attention weights, then
    classifies the aggregated representation.

    Architecture:
      1. Instance extractor — a pretrained 3D CNN (MONAI ResNet-18) with its
         classification head removed, producing per-patch feature vectors.
      2. Attention mechanism — two-layer MLP (Linear → Tanh → Linear) that
         assigns a scalar importance weight to each patch.
      3. Bag aggregator — weighted sum of patch features (attention pooling).
      4. Classifier — linear layer mapping the bag representation to class logits.

    Input shape: (1, N, 1, D, H, W) — one bag of N patches, single channel.
    Output shape: (1, num_classes) — bag-level class logits.

    Pretrained weights: MedicalNet encoder via ``get_r3d18_monai_model``
    (see that function for details on the MedicalNet pretrained source).
    Attention and classifier weights are initialised randomly.

    Args:
        instance_extractor: A pretrained 3D CNN whose final classification layer
            will be stripped to serve as a patch feature extractor.
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        feature_dim: Dimensionality of the per-patch feature vector produced by
            ``instance_extractor`` after the final classification layer is removed
            (512 for ResNet-18).
    """

    def __init__(self, instance_extractor: nn.Module, num_classes: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Strip the final classification head from the extractor; retain all
        # preceding layers as the per-patch feature encoder.
        self.instance_extractor = nn.Sequential(*list(instance_extractor.children())[:-1])

        # Attention mechanism: learn a scalar importance weight per patch.
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Bag-level classifier operating on the attention-aggregated feature vector.
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both standard 5D input (B, C, D, H, W) and MIL 6D input (B, N, C, D, H, W).
        # A 5D input is treated as a single-instance bag by inserting N=1.
        if x.dim() == 5:
            x = x.unsqueeze(1)  # (B, C, D, H, W) → (B, 1, C, D, H, W)

        # Squeeze the batch dimension (assumed = 1) to process the bag as a batch
        # of N instances through the instance extractor.
        if x.shape[0] == 1:
            x = x.squeeze(0)  # (N, C, D, H, W)

        # 1. Extract per-patch features: H has shape (N, feature_dim)
        h = self.instance_extractor(x)
        h = h.view(h.size(0), -1)

        # 2. Compute attention weights and normalise via softmax.
        #    a has shape (N, 1) → transposed to (1, N) for matrix multiply.
        a = self.attention(h)
        a_w = torch.softmax(a.t(), dim=1)  # (1, N)

        # 3. Aggregate: weighted sum of instance features → bag vector (1, feature_dim)
        m = torch.mm(a_w, h)

        # 4. Classify the bag → (1, num_classes)
        return self.classifier(m)


# ── 7. Model Factory ───────────────────────────────────────────────────────────

def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Construct and return a model by name.

    Model names are matched case-insensitively. Every name listed in
    ``SUPPORTED_MODELS`` is guaranteed to return a model that accepts a
    ``(B, 1, 128, 128, 128)`` input tensor and produces ``(B, num_classes)``
    logits on CPU.

    Args:
        model_name: Architecture name. Must be one of ``SUPPORTED_MODELS``
            (case-insensitive). Examples: ``"R3D18"``, ``"r3d18"``, ``"r3d50"``.
        num_classes: Number of output classes (typically 2: healthy / aneurysm).

    Returns:
        A ``torch.nn.Module`` ready for training or evaluation.

    Raises:
        ValueError: If ``model_name`` is not in ``SUPPORTED_MODELS``.
    """
    name = model_name.lower()

    if name == "r3d18":
        print("  -> Using torchvision R3D-18 (Kinetics-400 pretrained).")
        return get_r3d18_pytorch_model(num_classes)
    elif name == "r3d50":
        print("  -> Using MONAI ResNet-50 (MedicalNet pretrained).")
        return get_r3d50_model(num_classes)
    elif name == "densenet121":
        print("  -> Using MONAI DenseNet-121 (from scratch).")
        return get_densenet121_monai_model(num_classes)
    elif name == "swinunetr":
        print("  -> Using MONAI SwinUNETR classifier.")
        return get_swinunetr_model(num_classes)
    elif name == "unet3d":
        print("  -> Using custom UNet3D classifier (from scratch).")
        return UNet3DClassifier(in_channels=1, num_classes=num_classes, base_c=32)
    elif name == "unet3dwithbackbone":
        print("  -> Using custom UNet3D with MONAI ResNet-18 backbone (MedicalNet).")
        return get_unet3d_with_backbone_model(num_classes)
    elif name == "mil_r3d18":
        print("  -> Using MIL classifier with MONAI ResNet-18 instance extractor.")
        instance_extractor = get_r3d18_monai_model(num_classes)
        # ResNet-18 produces 512-dimensional features before the classification head.
        return MILClassifier(instance_extractor, num_classes, feature_dim=512)  # type: ignore
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Valid names (case-insensitive): {SUPPORTED_MODELS}"
        )
