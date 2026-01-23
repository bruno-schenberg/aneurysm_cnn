import torch
import torch.nn as nn
import torchvision.models.video as video_models
from typing import Optional

# ----------------------------------------------------
# 1. R3D-18 Model Definition (Existing)
# ----------------------------------------------------

def get_r3d18_model(num_classes: int, weights: Optional[str] = 'default') -> torch.nn.Module:
    """Loads and adapts R3D-18 for single-channel 3D classification."""
    if weights == 'default':
        model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
    else:
        model = video_models.r3d_18(weights=None)

    original_conv1 = model.stem[0]
    new_conv1 = nn.Conv3d(
        in_channels=1, # Single channel input (C=1)
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
    model.stem[0] = new_conv1

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# --- Helper Block for UNet3D ---
class ConvBlock3D(nn.Module):
    """A standard convolution block used in UNet."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------------------------------
# 2. UNet3D Model Definition (New)
# ----------------------------------------------------

class UNet3DClassifier(nn.Module):
    """
    A simplified 3D UNet-like architecture adapted for Classification.
    It uses the UNet encoder path and replaces the decoder/final output with
    Global Average Pooling (GAP) and a fully connected layer.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_c: int = 32):
        super().__init__()

        # Encoder Path (Downsampling)
        self.enc1 = ConvBlock3D(in_channels, base_c) # 128
        self.pool1 = nn.MaxPool3d(2) # 64
        self.enc2 = ConvBlock3D(base_c, base_c * 2) # 64
        self.pool2 = nn.MaxPool3d(2) # 32
        self.enc3 = ConvBlock3D(base_c * 2, base_c * 4) # 32
        self.pool3 = nn.MaxPool3d(2) # 16
        self.enc4 = ConvBlock3D(base_c * 4, base_c * 8) # 16
        self.pool4 = nn.MaxPool3d(2) # 8

        # Bottleneck (or deep features)
        self.bottleneck = ConvBlock3D(base_c * 8, base_c * 16) # 8x8x8 features

        # Classification Head
        # Use Global Average Pooling to reduce 8x8x8 to 1x1x1
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # The feature size after GAP is base_c * 16 (e.g., 512)
        self.classifier = nn.Sequential(
            nn.Linear(base_c * 16, num_classes)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Classification
        pooled = self.global_pool(b)
        # Flatten for the linear layer (B, C, 1, 1, 1) -> (B, C)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)

        return output

# ----------------------------------------------------
# 3. Universal Model Getter (Modified)
# ----------------------------------------------------

def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Selects and returns the specified model.
    """
    if model_name.lower() == 'r3d18':
        return get_r3d18_model(num_classes)
    elif model_name.lower() == 'unet3d':
        # Using 32 as the base channel count, a common starting point
        return UNet3DClassifier(in_channels=1, num_classes=num_classes, base_c=32)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'R3D18' or 'UNet3D'.")