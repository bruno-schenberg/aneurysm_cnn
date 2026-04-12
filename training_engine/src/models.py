"""
models.py

Model registry for the aneurysm CNN training engine. Provides a factory
function (``get_model``) that constructs any supported architecture by name.

## Common interface contract

Every architecture in ``SUPPORTED_MODELS`` accepts a single-channel 128³
input tensor of shape ``(B, 1, 128, 128, 128)`` and produces raw class logits
of shape ``(B, num_classes)``. The logits are passed to ``CrossEntropyLoss``
in the training loop — no softmax is applied inside the model.

## Transfer learning strategy

Training a 3D CNN from scratch on a small aneurysm dataset (typically a few
hundred cases) risks severe overfitting: the model has millions of parameters
but too few examples to constrain them. Transfer learning addresses this by
initialising the network with weights learned on a large dataset, then
fine-tuning on the target task. The pretrained weights encode general-purpose
feature detectors (edges, textures, shapes) that are useful across many tasks,
so the model needs to learn much less from the small dataset.

Two pretrained sources are used:

  **Kinetics-400 (torchvision R3D-18):** a large video action-recognition
  dataset (~300k clips). Although video data is very different from medical
  imaging, the spatial 3D convolution filters still learn useful low-level
  features (edges, gradients, local structure). The key advantage is dataset
  size and filter diversity.

  **MedicalNet / Med3D (MONAI ResNet-18 and ResNet-50):** a ResNet pretrained
  on 23 segmentation datasets covering brain, abdomen, cardiac, and other
  anatomical regions — approximately 2.4k labelled volumes in total. The
  features are learned directly on 3D medical images, which share the same
  intensity characteristics, noise properties, and anatomical structure as
  the aneurysm volumes. This is the more principled choice for medical imaging.

## Single-channel adaptation

Pretrained 3D CNNs expect RGB-like multi-channel input. The aneurysm volumes
have a single intensity channel. Two adaptation strategies are used:

  **Channel averaging (torchvision R3D-18):** the three pretrained input
  filters are averaged along the channel dimension to produce one filter.
  This preserves the spatial frequency selectivity learned during pretraining
  rather than discarding it. Summing instead of averaging would triple the
  activation magnitudes, requiring the subsequent batch-norm layers to
  re-calibrate; averaging keeps the magnitude scale consistent.

  **Native single-channel (MONAI models):** MONAI's ResNet, DenseNet, and
  SwinUNETR constructors accept ``n_input_channels=1`` directly. The first
  conv layer is created with the correct number of channels from the start,
  and the pretrained weights (where available) are also single-channel.

## Pretrained weight sources

  - Kinetics-400: downloaded by torchvision on first use.
  - MedicalNet: downloaded from ``TencentMedicalNet/MedicalNet-Resnet18`` and
    ``TencentMedicalNet/MedicalNet-Resnet50`` on Hugging Face Hub by MONAI on
    first use. Requires MONAI >= 1.3.1.
  - BTCV (SwinUNETR): loaded from a local cache file at
    ``/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth``.
    Must be downloaded manually.
  - DenseNet-121, UNet3D: no pretrained weights — trained from scratch.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from monai.networks.nets import DenseNet, SwinUNETR, UNETR, resnet10, resnet18, resnet50


# ── Model Registry ─────────────────────────────────────────────────────────────

SUPPORTED_MODELS: list[str] = [
    "R3D10",            # MONAI ResNet-10   — MedicalNet pretrained (23 medical datasets), lightest CNN
    "R3D18",            # torchvision R3D-18 — Kinetics-400 pretrained, adapted to 1 channel
    "R3D50",            # MONAI ResNet-50   — MedicalNet pretrained (23 medical datasets)
    "UNETR",            # MONAI UNETR — pure ViT encoder + CNN decoder, trained from scratch
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

    R3D-18 is an 18-layer 3D ResNet from torchvision's video model zoo. Each
    convolution operates on three spatial dimensions (D, H, W) simultaneously,
    making it naturally suited for volumetric inputs. Pretrained on Kinetics-400
    (RGB video, 3 channels), it must be adapted to accept single-channel medical
    volumes.

    **Single-channel adaptation — channel averaging:**
    The pretrained first conv layer has shape
    ``(out_channels, 3, kD, kH, kW)`` — three input channels for RGB. We
    construct a new conv layer with one input channel and initialise its weight
    as ``mean(pretrained_weights, dim=1, keepdim=True)``. Averaging rather than
    summing keeps the activation scale consistent: the new single-channel filter
    responds to grayscale intensity the same way the original RGB filters
    responded to colour, without inflating the output magnitude.

    **Classification head replacement:**
    The original Kinetics-400 head (400 action classes) is replaced with a
    fresh ``nn.Linear(in_features, num_classes)`` layer, initialised randomly.
    Only this head is newly initialised; all other layers start from pretrained
    weights and are fine-tuned during training.

    **Why Kinetics-400?**
    Despite the domain gap between video clips and MRI/CTA volumes, the learned
    3D spatial filters encode useful low-level structure: local gradients,
    curvature, and texture patterns that generalise to volumetric medical data.
    The large pretraining dataset (300k clips) provides much better filter
    diversity than random initialisation on a small medical dataset.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        weights: ``'default'`` loads Kinetics-400 pretrained weights. Any other
            value initialises the model with random weights (useful for ablation
            experiments that isolate the effect of pretraining).
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

    Uses MONAI's implementation of ResNet-18 with 3D convolutions throughout.
    Unlike the torchvision variant, MONAI's ResNet accepts ``n_input_channels=1``
    natively, so no channel adaptation is required.

    **MedicalNet pretraining:**
    Weights are sourced from the MedicalNet project (Tencent), a ResNet
    pre-trained on 23 medical imaging segmentation datasets spanning brain MRI,
    abdominal CT, cardiac MRI, and others — approximately 2.4k labelled 3D
    volumes. This is the most relevant pretraining source for the aneurysm task:
    the model has already learned to detect anatomical structures in 3D medical
    images before any fine-tuning.

    Weights are downloaded automatically from Hugging Face Hub
    (``TencentMedicalNet/MedicalNet-Resnet18``) on first use. If the download
    fails (offline environment, rate limit), the model falls back to random
    initialisation so training can proceed without crashing.

    **Classification head:**
    The original MedicalNet head is a 400-class linear layer (matching the
    number of segmentation categories across the pretraining datasets). It is
    replaced with a fresh linear layer sized for the binary aneurysm task.
    The ResNet-18 encoder produces 512-dimensional feature vectors.

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


# ── 2. R3D-10 Model ───────────────────────────────────────────────────────────

def get_r3d10_model(num_classes: int) -> torch.nn.Module:
    """
    MONAI ResNet-10 for single-channel 3D medical image classification.

    ResNet-10 is the lightest variant in the MedicalNet family: one residual
    block per stage instead of ResNet-18's two, giving roughly half the
    parameter count (~6M vs ~11M). Despite its size, it uses the same layer
    structure (layer1–layer4) and produces the same 512-dimensional feature
    vector before the classification head.

    **Why ResNet-10 is worth testing:**
    On small datasets (a few hundred cases), a model with fewer parameters is
    less likely to overfit. The capacity reduction also means faster training
    and lower GPU memory usage. Prior aneurysm detection literature frequently
    uses ResNet-10 as the primary architecture precisely for this reason —
    it often matches or outperforms larger models when labelled data is scarce.

    **MedicalNet pretraining:**
    Same source as ResNet-18 and ResNet-50 — the MedicalNet project (Tencent),
    pre-trained on 23 medical imaging datasets. Weights are downloaded from
    ``TencentMedicalNet/MedicalNet-Resnet10`` on Hugging Face Hub on first use,
    with the same offline fallback behaviour as the other MedicalNet variants.

    The 400-class head is replaced with a fresh linear layer sized for the
    binary aneurysm task. The encoder produces 512-dimensional features.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
    """
    try:
        model = resnet10(pretrained=True, spatial_dims=3, n_input_channels=1, num_classes=400)
    except Exception:
        model = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=400)

    # ResNet-10 encoder produces 512-dimensional features, same as ResNet-18.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ── 3. R3D-50 Model ───────────────────────────────────────────────────────────

def get_r3d50_model(num_classes: int) -> torch.nn.Module:
    """
    MONAI ResNet-50 for single-channel 3D medical image classification.

    ResNet-50 uses bottleneck residual blocks (1×1 → 3×3 → 1×1 convolutions)
    instead of the plain 3×3 → 3×3 blocks in ResNet-18. This increases model
    capacity (25M vs 11M parameters) and the encoder feature dimension (2048
    vs 512), at the cost of higher memory usage and longer training time.

    **MedicalNet pretraining:**
    Identical source to the ResNet-18 variant — the MedicalNet project
    (Tencent), pre-trained on the same 23 medical imaging datasets. Weights are
    downloaded from ``TencentMedicalNet/MedicalNet-Resnet50`` on Hugging Face
    Hub on first use, with the same offline fallback behaviour.

    The original 400-class head is replaced with a fresh linear layer sized for
    the binary aneurysm task. ResNet-50's bottleneck encoder produces
    2048-dimensional features before the classification head.

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

    DenseNet differs from ResNet in its connectivity pattern: every layer
    receives feature maps from *all* preceding layers in the dense block, not
    just the immediately previous one. This dense reuse of earlier features
    encourages the network to learn complementary representations at each layer
    rather than redundant ones, and provides strong gradient flow to early
    layers without requiring residual shortcuts.

    DenseNet-121 has four dense blocks with 6, 12, 24, and 16 layers
    respectively, separated by transition layers that halve spatial resolution.
    The growth rate (32) controls how many new feature maps each layer adds to
    the collective representation; the initial feature count is 64.

    **No pretrained weights:**
    No MedicalNet or Kinetics-400 weights are available for 3D DenseNet-121,
    so this model is trained entirely from scratch on the aneurysm dataset.
    Despite the lack of pretraining, DenseNet's feature reuse architecture
    provides some regularisation that mitigates overfitting on small datasets.

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

    SwinUNETR is a hierarchical Swin Transformer architecture originally
    designed for 3D medical image segmentation. A Swin Transformer partitions
    the volume into non-overlapping windows and applies self-attention within
    each window, with a shifting window mechanism between layers to allow
    cross-window information exchange. The hierarchical design produces
    multi-scale feature maps at four resolutions.

    **Encoder-only adaptation:**
    The full SwinUNETR is constructed to load the pretrained checkpoint
    (which covers the encoder sub-module ``swinViT``), but the decoder and
    segmentation head are discarded. Only the deepest encoder output is used
    for classification. This repurposes the powerful Transformer feature
    extractor for a classification task without requiring segmentation labels.

    **Classification head:**
    The deepest encoder output has shape
    ``(B, feature_size×16, D/32, H/32, W/32)`` — for ``feature_size=48``
    and a 128³ input this is ``(B, 768, 4, 4, 4)``. Global Average Pooling
    collapses the spatial dimensions to ``(B, 768)``, which is then mapped to
    class logits by a linear layer. GAP is preferred over flattening because
    it is spatially invariant and produces a fixed-length vector regardless of
    input size.

    **BTCV pretraining:**
    The Swin Transformer encoder (``swinViT``) is loaded from a checkpoint
    trained on the BTCV (Beyond the Cranial Vault) multi-organ segmentation
    challenge — 14 abdominal organ classes. The weights are read from a local
    cache file at ``/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth``.
    If the file is absent, the model falls back to random initialisation and
    prints a warning — construction and the forward pass still succeed.

    **Note on MONAI >= 1.4:**
    Earlier MONAI versions required an ``img_size`` constructor argument to
    pre-compute positional encodings. From MONAI >= 1.4, the window-based
    attention is position-independent and accepts inputs of any spatial size.
    The ``img_size`` parameter is kept in the constructor signature for API
    compatibility but is not forwarded to SwinUNETR.

    **Dependency:** requires ``einops`` (``pip install einops``) for the
    Swin Transformer window-partitioning operations.

    Args:
        img_size: Input spatial size. Kept for API compatibility; not passed
            to SwinUNETR internally (see MONAI >= 1.4 note above).
        in_channels: Number of input channels (1 for single-channel MRI/CTA).
        num_classes: Number of output classes (typically 2).
        feature_size: Base feature dimensionality for the Swin Transformer.
            Must match the pretrained checkpoint (48 for the BTCV checkpoint).
        use_pretrained: If ``True``, attempts to load the BTCV encoder weights.
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


# ── 5. UNETR Model ────────────────────────────────────────────────────────────

class UNETRClassifier(nn.Module):
    """
    UNETR encoder repurposed for 3D volumetric classification.

    UNETR (UNet TRansformer) uses a pure Vision Transformer (ViT) as encoder
    and a CNN decoder for segmentation. Here only the ViT encoder is used —
    the decoder is discarded and replaced with mean pooling over patch tokens
    followed by a linear classification head.

    **How the ViT encoder works:**
    The input volume is divided into non-overlapping 3D patches of size
    16×16×16. Each patch is flattened and projected to a ``hidden_size``-
    dimensional embedding. The 512 patch tokens (for a 128³ input) are then
    processed by a standard multi-head self-attention Transformer, where every
    token attends to every other token globally — there is no windowing or
    locality constraint. This gives the model a global receptive field from the
    first layer, unlike CNNs which build up receptive field gradually.

    **UNETR vs SwinUNETR:**
    SwinUNETR uses hierarchical windowed attention (Swin Transformer), which
    restricts attention to local windows and progressively merges them. This
    reduces compute but loses global context at early layers. UNETR's pure ViT
    attends globally from the start, which can capture long-range anatomical
    relationships but is more expensive (O(N²) attention over 512 tokens per
    layer). On a small dataset, global attention may be harder to learn than
    the locally-biased Swin attention.

    **Classification head:**
    The ViT forward pass returns a tensor of shape ``(B, N_patches, hidden_size)``
    — one embedding per patch token. Mean pooling over the patch dimension
    collapses this to ``(B, hidden_size)``, treating all patches equally as
    contributors to the scan-level representation. A single linear layer maps
    this to class logits. The ``[CLS]`` token approach used in image
    classification ViTs is not used here because MONAI's UNETR ViT is
    configured with ``classification=False``, which omits the CLS token.

    **No pretrained weights:**
    Unlike SwinUNETR, no publicly available pretrained UNETR checkpoint is
    loaded here. The model trains entirely from scratch. MONAI's model zoo
    does include pretrained UNETR variants (e.g. BTCV segmentation), but
    loading them requires matching the exact architecture hyperparameters and
    manually mapping weight keys — which is brittle across MONAI versions.
    From scratch, this model primarily tests the architectural hypothesis
    (global ViT attention vs local CNN filters) rather than the pretraining
    benefit.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        img_size: Input spatial size. Must match the actual input; used by
            the ViT to compute positional embeddings.
        hidden_size: Transformer embedding dimension. 768 = ViT-Base config.
        mlp_dim: Feed-forward hidden dimension inside each Transformer block.
            3072 = 4 × hidden_size (standard ViT-Base ratio).
        num_heads: Number of attention heads. Must divide ``hidden_size``.
            12 = ViT-Base config.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: tuple = (128, 128, 128),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Instantiate the full UNETR to access the pretrained ViT encoder sub-module.
        # out_channels is arbitrary here since the decoder is not used.
        self.unetr = UNETR(
            in_channels=1,
            out_channels=14,
            img_size=img_size,
            feature_size=16,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
        )

        # Classification head operating on the mean-pooled patch tokens.
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MONAI ViT returns (x, hidden_states_out) where x has shape
        # (B, N_patches, hidden_size). For a 128³ input with patch_size=16:
        # N_patches = (128/16)³ = 512 patch tokens.
        x_enc, _ = self.unetr.vit(x)

        # Mean pool over patch tokens to get a fixed-length scan representation.
        pooled = x_enc.mean(dim=1)   # (B, hidden_size)
        return self.classifier(pooled)


def get_unetr_model(
    num_classes: int,
    img_size: tuple = (128, 128, 128),
) -> torch.nn.Module:
    """
    Constructs a ``UNETRClassifier`` with ViT-Base hyperparameters.

    See ``UNETRClassifier`` for full architecture details and the scientific
    rationale for including a pure ViT alongside the hierarchical SwinUNETR.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        img_size: Spatial size of the input volumes. Must match the actual
            input shape because the ViT positional embeddings are constructed
            for exactly this number of patches (``img_size / patch_size`` per
            axis). Defaults to ``(128, 128, 128)``; pass ``(256, 256, 128)``
            when using the 256×256×128 dataset variants.
    """
    return UNETRClassifier(
        num_classes=num_classes,
        img_size=img_size,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
    )


# ── 6. UNet3D Models ──────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """
    Standard double-convolution block used in the UNet encoder and decoder paths.

    Each block applies two sequential 3×3×3 convolutions, each followed by
    Batch Normalisation and ReLU activation. This is the canonical UNet building
    block as defined in Ronneberger et al. (2015), extended to 3D.

    **Why two convolutions per block?**
    A single convolution has a receptive field of 3×3×3 voxels. Two stacked
    convolutions have an effective receptive field of 5×5×5 while using fewer
    parameters than a single 5×5×5 convolution — and the non-linearity between
    them increases representational capacity.

    **Batch Normalisation placement:**
    BatchNorm is applied after each convolution and before ReLU, following the
    convention established in the original ResNet paper. BatchNorm normalises
    the pre-activation distribution, which stabilises training and reduces
    sensitivity to weight initialisation.

    **``inplace=True`` on ReLU:**
    Saves memory by overwriting the input tensor rather than allocating a new
    one. Safe here because the ReLU output is not needed for any backward pass
    other than through the ReLU itself.

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
    Custom 3D UNet adapted for volumetric classification instead of segmentation.

    The classic UNet architecture has a symmetric encoder-decoder structure with
    skip connections that concatenate encoder feature maps into the decoder at
    each scale. Originally designed for pixel-wise segmentation, it is adapted
    here for scan-level classification by replacing the final segmentation
    output (per-voxel class scores) with Global Average Pooling (GAP) and a
    linear classification head.

    **Why UNet for classification?**
    The skip connections force the network to maintain both local detail
    (fine-grained texture and edge information from early encoder layers) and
    global context (semantic features from the bottleneck) throughout the
    forward pass. This multi-scale feature hierarchy can be beneficial for
    detecting small structures like aneurysms, which occupy a tiny fraction of
    the total volume but require context from the surrounding vasculature for
    confident classification.

    **Architecture (with ``base_c=32``):**
    Encoder:  128³→64³→32³→16³→8³ (channel counts: 32, 64, 128, 256)
    Bottleneck:  8³ (512 channels)
    Decoder:  8³→16³→32³→64³→128³ (channel counts: 256, 128, 64, 32)
    Head: GAP → Linear(32, num_classes)

    Skip connections concatenate the corresponding encoder map to the
    upsampled decoder map before each decoder ConvBlock, doubling the input
    channels at each decoder stage.

    **Global Average Pooling vs flattening:**
    GAP reduces the final ``(B, base_c, 128, 128, 128)`` feature map to
    ``(B, base_c)`` by averaging over spatial dimensions. Flattening would
    produce a ``(B, base_c × 128³)`` vector — over 67M elements — which is
    completely impractical. GAP also provides spatial invariance: the
    classification score is based on the mean activation across all locations,
    not on where in the volume a feature appears.

    **No pretrained weights:** trained from scratch on the aneurysm dataset.

    Args:
        in_channels: Number of input channels (1 for single-channel MRI/CTA).
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        base_c: Base channel count. Channel counts at each encoder/decoder level
            are ``base_c × 2^level``. Increasing ``base_c`` raises model
            capacity and memory cost.
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

    Combines the skip-connection structure of UNet with the medically pretrained
    feature extractors of MedicalNet. The ResNet-18 encoder replaces the plain
    ConvBlock3D encoder path; its four residual-block stages (layer1–layer4)
    serve as the skip-connection sources for a standard UNet decoder. The final
    decoded feature map is global-average-pooled and classified.

    **Why combine UNet and a pretrained backbone?**
    A plain ResNet (R3D18, R3D50) fine-tuned for classification uses only the
    deepest encoder features. UNet's skip connections reintroduce fine-grained
    spatial detail from earlier encoder stages into the classification decision.
    For aneurysm detection — where the lesion is small and its boundary detail
    matters — this multi-scale integration may help the model distinguish true
    aneurysms from nearby vascular structures that look similar at coarse scales.

    **Encoder channel sizes (ResNet-18):**
    stem output (e0): 64 channels
    layer1 (e1): 64 channels  (16 × 16 × 16 spatial)  -- note: no spatial reduction from layer1 in MONAI ResNet with stride=1 in layer1
    layer2 (e2): 128 channels
    layer3 (e3): 256 channels
    layer4 / bottleneck: 512 channels

    The decoder up-samples from the bottleneck back to the stem resolution,
    concatenating the corresponding encoder feature map at each scale.

    **MONAI ResNet activation naming:**
    MONAI's ResNet uses ``.act`` for the stem activation (not ``.relu`` as in
    torchvision). The forward method accesses internal ResNet attributes
    (``conv1``, ``bn1``, ``act``, ``maxpool``, ``layer1``–``layer4``) directly.
    This is fragile with respect to MONAI internal API changes.

    Args:
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        base_c: Base channel count for the decoder stages.
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


# ── 7. Multiple Instance Learning (MIL) Model ─────────────────────────────────

class MILClassifier(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) classifier for 3D volumes.

    **The MIL paradigm:**
    In standard supervised learning, each scan is a single labelled example.
    In MIL, each scan is a *bag* containing multiple unlabelled *instances*
    (patches). The bag label (aneurysm present / absent) is known, but the
    per-patch labels are not. A bag is positive if at least one instance is
    positive. The model must learn which patches are diagnostically relevant
    and aggregate their evidence into a bag-level prediction.

    **Why MIL for aneurysm detection?**
    An aneurysm typically occupies a small region of the full 128³ volume.
    A global classifier operating on the whole scan may not learn to focus on
    the relevant region — it sees too much background. MIL forces the model
    to learn patch-level discriminative features and identify which patches
    contribute to the positive label, providing an implicit form of spatial
    attention over the volume.

    **Architecture:**
    1. *Instance extractor* — a pretrained MONAI ResNet-18 with its final
       linear layer removed. Each patch of shape ``(1, D, H, W)`` is encoded
       independently to a ``feature_dim``-dimensional vector. In practice
       ``feature_dim=512`` for ResNet-18.

    2. *Attention mechanism* — a two-layer MLP (Linear(512→128) → Tanh →
       Linear(128→1)) that assigns a scalar logit to each patch. Softmax
       normalises the logits into attention weights summing to 1 across the
       bag. Tanh in the hidden layer allows the attention to model both
       excitatory and inhibitory evidence, unlike ReLU which is one-sided.

    3. *Bag aggregator* — weighted sum of patch features:
       ``M = A^T × H`` where ``H`` is (N, feature_dim) and ``A`` is (1, N).
       This produces a single (1, feature_dim) bag representation that
       concentrates information from the most diagnostically relevant patches.

    4. *Classifier* — Linear(feature_dim, num_classes) mapping the bag
       representation to class logits.

    **Batch size constraint:**
    This implementation assumes batch size = 1 (one bag per forward pass).
    The ``forward`` method squeezes the batch dimension to process the N
    patches as a batch through the instance extractor. Passing batch size > 1
    will produce incorrect results because the squeeze at
    ``if x.shape[0] == 1`` will not fire, leaving a 6D tensor that the
    extractor cannot process.

    **Head stripping:**
    ``nn.Sequential(*list(instance_extractor.children())[:-1])`` removes the
    last child module (the ``fc`` layer) from the ResNet to expose the pooled
    feature vector. This relies on ``children()`` returning a flat list in the
    correct forward order, which holds for MONAI ResNet but is fragile with
    respect to architectures that have special forward logic not reflected in
    their module hierarchy.

    Args:
        instance_extractor: A pretrained 3D CNN whose final classification
            layer will be stripped. Must produce a spatial feature map that
            flattens to ``feature_dim`` after ``view``.
        num_classes: Number of output classes (typically 2).
        feature_dim: Dimensionality of the per-patch feature vector after the
            final classification layer is removed (512 for ResNet-18).
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


# ── 7. Multimodal Late-Fusion Wrapper ─────────────────────────────────────────

def _strip_classifier(model: nn.Module, model_name: str) -> int:
    """
    Replaces the final classification linear layer of ``model`` with an Identity
    module so that the model's forward pass returns feature vectors instead of
    class logits. Returns the feature dimensionality (the replaced layer's
    ``in_features``).

    Used internally to prepare an image-only model for late fusion with tabular
    features via ``MultiModalWrapper``.

    Args:
        model: An already-constructed image model whose final linear layer will
            be replaced.
        model_name: Architecture name (case-insensitive), used to locate the
            correct attribute path for each architecture.

    Returns:
        Integer feature dimensionality of the stripped classification layer.

    Raises:
        ValueError: If ``model_name`` is not a recognised architecture.
    """
    name = model_name.lower()
    if name in ("r3d18", "r3d50"):
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return dim
    elif name == "densenet121":
        # MONAI DenseNet: class_layers is Sequential(relu, pool, flatten, out).
        # Replacing 'out' leaves relu+pool+flatten intact so the output is
        # the flattened pooled feature vector, not logits.
        dim = model.class_layers.out.in_features
        model.class_layers.out = nn.Identity()
        return dim
    elif name in ("swinunetr", "unet3d", "unet3dwithbackbone", "mil_r3d18"):
        dim = model.classifier.in_features
        model.classifier = nn.Identity()
        return dim
    else:
        raise ValueError(
            f"_strip_classifier is not implemented for model '{model_name}'. "
            f"Valid names: {SUPPORTED_MODELS}"
        )


class MultiModalWrapper(nn.Module):
    """
    Late-fusion multimodal wrapper that combines image features from any CNN
    backbone with tabular patient metadata (age, gender) for joint classification.

    Architecture:
      - Image branch: the provided ``image_model`` with its classifier stripped,
        producing a feature vector of shape ``(B, image_feature_dim)``.
      - Tabular branch: a small MLP that embeds ``[age_normalised, gender]``
        into a 16-dimensional vector.
      - Fusion head: concatenates both branches and passes through a dropout MLP
        to produce class logits.

    The tabular branch is intentionally shallow (two linear layers) to avoid
    overfitting on only two input features.

    Input:
        image: ``(B, 1, D, H, W)`` volumetric scan tensor.
        tabular: ``(B, tabular_dim)`` float tensor — by default ``[age/100, gender]``.

    Output:
        ``(B, num_classes)`` class logits.

    Args:
        image_model: A CNN backbone whose final classifier has already been
            stripped by ``_strip_classifier``.
        image_feature_dim: Dimensionality of the feature vector produced by
            ``image_model`` after stripping.
        tabular_dim: Number of tabular input features (default 2: age + gender).
        num_classes: Number of output classes.
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
        img_feat = self.image_model(image)       # (B, image_feature_dim)
        tab_feat = self.tabular_branch(tabular)  # (B, 16)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.fusion_head(fused)


# ── 8. Model Factory ───────────────────────────────────────────────────────────

def get_model(
    model_name: str,
    num_classes: int,
    use_tabular: bool = False,
    tabular_dim: int = 2,
    spatial_size: tuple = (128, 128, 128),
) -> torch.nn.Module:
    """
    Construct and return a model by name.

    This is the single entry point used by the orchestrator to instantiate any
    supported architecture. Using a factory function decouples the training
    pipeline from specific model classes: adding a new architecture requires
    only a new branch here and a new entry in ``SUPPORTED_MODELS``, with no
    changes to the orchestrator or training loop.

    Model names are matched case-insensitively so ``"r3d18"``, ``"R3D18"``,
    and ``"R3d18"`` all produce the same model.

    Every returned model:
      - Accepts input of shape ``(B, 1, *spatial_size)``
      - Returns raw logits of shape ``(B, num_classes)``
      - Is on CPU; the caller is responsible for ``.to(device)``

    When ``use_tabular=True``, the model's classification head is stripped and
    it is wrapped in a ``MultiModalWrapper`` that accepts an additional
    ``tabular`` tensor of shape ``(B, tabular_dim)`` alongside the image.
    The wrapper's ``forward`` signature becomes ``forward(image, tabular)``.

    Args:
        model_name: Architecture name. Must be one of ``SUPPORTED_MODELS``
            (case-insensitive). Examples: ``"R3D18"``, ``"R3D50"``.
        num_classes: Number of output classes (typically 2: healthy / aneurysm).
        use_tabular: If True, wrap the model for late fusion with tabular features.
        tabular_dim: Number of tabular input features (default 2: age + gender).
        spatial_size: Input spatial dimensions as ``(H, W, D)``. Forwarded to
            UNETR to set positional embedding size; CNN-based models ignore it
            because they use adaptive average pooling.

    Returns:
        A ``torch.nn.Module`` ready for ``.to(device)`` and training.

    Raises:
        ValueError: If ``model_name`` does not match any entry in
            ``SUPPORTED_MODELS``.
    """
    name = model_name.lower()

    if name == "r3d10":
        print("  -> Using MONAI ResNet-10 (MedicalNet pretrained).")
        return get_r3d10_model(num_classes)
    elif name == "r3d18":
        print("  -> Using torchvision R3D-18 (Kinetics-400 pretrained).")
        model = get_r3d18_pytorch_model(num_classes)
    elif name == "r3d50":
        print("  -> Using MONAI ResNet-50 (MedicalNet pretrained).")
        model = get_r3d50_model(num_classes)
    elif name == "unetr":
        print("  -> Using MONAI UNETR (pure ViT encoder, from scratch).")
        return get_unetr_model(num_classes, img_size=spatial_size)
    elif name == "densenet121":
        print("  -> Using MONAI DenseNet-121 (from scratch).")
        model = get_densenet121_monai_model(num_classes)
    elif name == "swinunetr":
        print("  -> Using MONAI SwinUNETR classifier.")
        model = get_swinunetr_model(num_classes)
    elif name == "unet3d":
        print("  -> Using custom UNet3D classifier (from scratch).")
        model = UNet3DClassifier(in_channels=1, num_classes=num_classes, base_c=32)
    elif name == "unet3dwithbackbone":
        print("  -> Using custom UNet3D with MONAI ResNet-18 backbone (MedicalNet).")
        model = get_unet3d_with_backbone_model(num_classes)
    elif name == "mil_r3d18":
        print("  -> Using MIL classifier with MONAI ResNet-18 instance extractor.")
        instance_extractor = get_r3d18_monai_model(num_classes)
        # ResNet-18 produces 512-dimensional features before the classification head.
        model = MILClassifier(instance_extractor, num_classes, feature_dim=512)  # type: ignore
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Valid names (case-insensitive): {SUPPORTED_MODELS}"
        )

    if use_tabular:
        print(f"  -> Wrapping with MultiModalWrapper (tabular_dim={tabular_dim}).")
        feature_dim = _strip_classifier(model, name)
        model = MultiModalWrapper(model, feature_dim, tabular_dim, num_classes)

    return model
