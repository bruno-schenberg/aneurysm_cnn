import torch
import torch.nn as nn
import torchvision.models.video as video_models
from monai.networks.nets import ResNet, DenseNet, SwinUNETR
from monai.networks.blocks import ResNetBlock, Bottleneck, get_conv_layer
from monai.apps import MedicalNet
from typing import Optional

# ----------------------------------------------------
# 1. R3D-18 Model Definitions
# ----------------------------------------------------

def get_r3d18_pytorch_model(num_classes: int, weights: Optional[str] = 'default') -> torch.nn.Module:
    """
    Loads and adapts the torchvision R3D-18 model (pre-trained on videos)
    for single-channel 3D classification.
    """
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

def get_r3d18_monai_model(num_classes: int) -> torch.nn.Module:
    """
    Loads a 3D ResNet-18 model from MONAI, pre-trained on the MedicalNet dataset.
    The final classification layer is replaced to match the required number of classes.
    """
    # Load the MedicalNet ResNet-18 with its pre-trained weights
    model = MedicalNet(
        resnet_depth=18,
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2, # The original model has 2 classes, we will replace this
        pretrained=True
    )

    # The MedicalNet's classification head is named 'fc'.
    # We replace it with a new linear layer for our specific number of classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def get_r3d50_pytorch_model(num_classes: int, weights: Optional[str] = 'default') -> torch.nn.Module:
    """
    Loads and adapts the torchvision Slow R50 model (pre-trained on videos)
    for single-channel 3D classification.
    """
    if weights == 'default':
        model = video_models.slow_r50(weights=video_models.Slow_R50_Weights.DEFAULT)
    else:
        model = video_models.slow_r50(weights=None)

    # The first conv layer in slow_r50 is in blocks[0].conv
    original_conv1 = model.blocks[0].conv
    new_conv1 = nn.Conv3d(
        in_channels=1, # Single channel input (C=1)
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False # slow_r50's first conv has no bias
    )
    # Average the weights of the original 3-channel conv layer
    new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
    model.blocks[0].conv = new_conv1

    # The classifier in slow_r50 is a linear layer in blocks[6].proj
    num_ftrs = model.blocks[6].proj.in_features
    model.blocks[6].proj = nn.Linear(num_ftrs, num_classes)

    return model

def get_r3d50_monai_model(num_classes: int) -> torch.nn.Module:
    """
    Loads a 3D ResNet-50 model from MONAI, pre-trained on the MedicalNet dataset.
    The final classification layer is replaced to match the required number of classes.
    """
    # Load the MedicalNet ResNet-50 with its pre-trained weights
    model = MedicalNet(
        resnet_depth=50,
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2, # The original model has 2 classes, we will replace this
        pretrained=True
    )

    # The MedicalNet's classification head is named 'fc'.
    # We replace it with a new linear layer for our specific number of classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# ----------------------------------------------------
# 2. DenseNet Model Definition
# ----------------------------------------------------

def get_densenet121_monai_model(num_classes: int) -> torch.nn.Module:
    """
    Loads a 3D DenseNet-121 model from MONAI, suitable for medical imaging.
    Note: This model is NOT pre-trained.
    """
    model = DenseNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        # The following parameters define the DenseNet-121 architecture
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16)
    )
    return model

# ----------------------------------------------------
# 3. SwinUNETR Model Definition
# ----------------------------------------------------

class SwinUNETRClassifier(nn.Module):
    """
    Wraps the SwinUNETR model to adapt it for a classification task.
    It uses the SwinUNETR as a feature extractor and adds a classification head.
    """
    def __init__(self, img_size: tuple, in_channels: int, num_classes: int, feature_size: int = 48, use_pretrained: bool = True):
        super().__init__()
        self.feature_size = feature_size

        # Instantiate SwinUNETR as the backbone
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=14,  # The pre-trained model was for 14-class segmentation
            feature_size=self.feature_size,
            use_checkpoint=True,
        )

        if use_pretrained:
            print("  -> Loading pre-trained weights for SwinUNETR backbone...")
            try:
                # These weights were trained on the BTCV dataset for multi-organ segmentation
                weights = torch.load("/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth")
                self.swin_unetr.load_from(weights)
                print("  -> Pre-trained weights loaded successfully.")
            except FileNotFoundError:
                print("  -> Warning: Pre-trained weights not found. The model will be trained from scratch.")
            except Exception as e:
                print(f"  -> Error loading pre-trained weights: {e}. The model will be trained from scratch.")

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        # Get the list of hidden states from the SwinUNETR encoder
        # The last element of the list is the final feature map from the encoder.
        hidden_states_out = self.swin_unetr(x, feature_only=True)
        encoder_output = hidden_states_out[-1]

        # Apply global average pooling to the encoder's output feature map
        pooled = self.global_pool(encoder_output)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten for the linear layer
        output = self.classifier(pooled)
        return output

def get_swinunetr_model(num_classes: int) -> torch.nn.Module:
    # The pre-trained model expects 96x96x96 inputs
    return SwinUNETRClassifier(img_size=(96, 96, 96), in_channels=1, num_classes=num_classes, use_pretrained=True)

# ----------------------------------------------------
# 4. UNet3D Model Definition
# ----------------------------------------------------

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

class UNet3DWithBackbone(nn.Module):
    """
    A 3D UNet-like architecture that uses a pre-trained ResNet as an encoder.
    It extracts features from the backbone for skip connections to a standard
    UNet decoder, and finishes with a classification head.
    """
    def __init__(self, num_classes: int = 2, base_c: int = 32):
        super().__init__()

        # --- Encoder (Pre-trained Backbone) ---
        # We use a ResNet-18 pre-trained on MedicalNet.
        # The backbone's layers will serve as our encoder.
        self.backbone = get_r3d18_monai_model(num_classes=num_classes)

        # --- Decoder Path (Upsampling) ---
        # The decoder channel sizes must match the output of the ResNet layers.
        # ResNet-18 layer outputs: layer1=64, layer2=128, layer3=256, layer4=512
        # The bottleneck will be the output of layer4.
        
        # Upsample from 512 (layer4) to 256 channels
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        # Concatenated: 256 (from up3) + 256 (from layer3) = 512
        self.dec3 = ConvBlock3D(512, 256)

        # Upsample from 256 to 128 channels
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        # Concatenated: 128 (from up2) + 128 (from layer2) = 256
        self.dec2 = ConvBlock3D(256, 128)

        # Upsample from 128 to 64 channels
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        # Concatenated: 64 (from up1) + 64 (from layer1) = 128
        self.dec1 = ConvBlock3D(128, 64)

        # Upsample from 64 to `base_c` channels
        self.up0 = nn.ConvTranspose3d(64, base_c, kernel_size=2, stride=2)
        # Concatenated: `base_c` (from up0) + 64 (from initial conv) = base_c + 64
        self.dec0 = ConvBlock3D(base_c + 64, base_c)

        # --- Classification Head ---
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(base_c, num_classes)

    def forward(self, x):
        # --- Encoder Path (using the backbone) ---
        # We need to run the backbone layer by layer to get intermediate features.
        
        # Initial convolution (e.g., 1 -> 64 channels)
        e0 = self.backbone.conv1(x)
        e0 = self.backbone.bn1(e0)
        e0 = self.backbone.relu(e0)
        
        # First downsampling and layer (e.g., 64 -> 64 channels)
        e1_pool = self.backbone.maxpool(e0)
        e1 = self.backbone.layer1(e1_pool)
        
        # Second layer (e.g., 64 -> 128 channels)
        e2 = self.backbone.layer2(e1)
        
        # Third layer (e.g., 128 -> 256 channels)
        e3 = self.backbone.layer3(e2)
        
        # Fourth layer (serves as the bottleneck)
        bottleneck = self.backbone.layer4(e3)

        # --- Decoder Path with Skip Connections ---
        d3 = self.up3(bottleneck)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat((e0, d0), dim=1)
        d0 = self.dec0(d0)

        # --- Classification Head ---
        pooled = self.global_pool(d0)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)

        return output

def get_unet3d_with_backbone_model(num_classes: int) -> torch.nn.Module:
    """Instantiates the UNet with a pre-trained ResNet-18 backbone."""
    return UNet3DWithBackbone(num_classes=num_classes, base_c=32)

class UNet3DClassifier(nn.Module):
    """
    A 3D UNet-like architecture adapted for Classification. It uses the full
    encoder-decoder path with skip connections, and replaces the final
    segmentation output with Global Average Pooling (GAP) and a fully
    connected layer for classification.    """
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

        # Decoder Path (Upsampling)
        self.up4 = nn.ConvTranspose3d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_c * 16, base_c * 8) # Concatenated: (base_c*8 from up4) + (base_c*8 from enc4)
        self.up3 = nn.ConvTranspose3d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_c * 8, base_c * 4) # Concatenated: (base_c*4) + (base_c*4)
        self.up2 = nn.ConvTranspose3d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_c * 4, base_c * 2) # Concatenated: (base_c*2) + (base_c*2)
        self.up1 = nn.ConvTranspose3d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_c * 2, base_c) # Concatenated: (base_c) + (base_c)

        # Classification Head
        # Use Global Average Pooling on the final decoder feature map
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # The feature size after GAP is `base_c` (e.g., 32)
        self.classifier = nn.Sequential(
            nn.Linear(base_c, num_classes)
        )

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(e4))

        # Decoder Path with Skip Connections
        d4 = self.up4(bottleneck)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # Classification Head
        pooled = self.global_pool(d1)
        # Flatten for the linear layer (B, C, 1, 1, 1) -> (B, C)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)

        return output

# ----------------------------------------------------
# 5. Multiple Instance Learning (MIL) Model
# ----------------------------------------------------

class MILClassifier(nn.Module):
    """
    A Multiple Instance Learning (MIL) classifier.

    This model uses a pre-trained 3D CNN as a feature extractor for instances (patches)
    and then uses an attention mechanism to aggregate instance features into a single
    bag-level representation for final classification.
    """
    def __init__(self, instance_extractor: nn.Module, num_classes: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # 1. Instance Feature Extractor
        # We remove the final classification layer from the provided extractor.
        # This assumes the extractor has a 'fc' attribute (like MedicalNet ResNets).
        self.instance_extractor = nn.Sequential(*list(instance_extractor.children())[:-1])

        # 2. Attention-based Aggregation
        # These layers will learn to assign an importance score (attention) to each instance.
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 3. Final Classifier
        # This classifier operates on the aggregated bag-level feature vector.
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        # x is the "bag" of instances, with shape (Batch, Num_Instances, C, D, H, W)
        # For a batch size of 1, this is (1, N, 1, 96, 96, 96)
        
        # Squeeze the batch dimension if it's 1, as we process instances of a single bag.
        if x.shape[0] == 1:
            x = x.squeeze(0) # Shape becomes (N, 1, 96, 96, 96)

        # 1. Get features for each instance
        # H has shape (N, feature_dim)
        H = self.instance_extractor(x)
        H = H.view(H.size(0), -1) # Flatten features

        # 2. Calculate attention scores for each instance
        # A has shape (N, 1)
        A = self.attention(H)
        # Apply softmax to get weights that sum to 1
        # A_w has shape (1, N)
        A_w = torch.softmax(A.t(), dim=1)

        # 3. Aggregate instance features using attention weights
        # M is the bag representation, shape (1, feature_dim)
        M = torch.mm(A_w, H)

        # 4. Classify the bag
        # logits has shape (1, num_classes)
        logits = self.classifier(M)

        return logits

# ----------------------------------------------------
# 6. Universal Model Getter
# ----------------------------------------------------

def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Selects and returns the specified model.
    """
    model_name_lower = model_name.lower()
    if model_name_lower in ('r3d18', 'r3d18_pytorch'):
        print("  -> Using torchvision R3D-18 model.")
        return get_r3d18_pytorch_model(num_classes)
    elif model_name_lower in ('r3d50', 'r3d50_pytorch'):
        print("  -> Using torchvision R3D-50 model.")
        return get_r3d50_pytorch_model(num_classes)
    elif model_name_lower == 'r3d18_monai':
        print("  -> Using MONAI ResNet-18 model.")
        return get_r3d18_monai_model(num_classes)
    elif model_name_lower == 'r3d50_monai':
        print("  -> Using MONAI ResNet-50 model.")
        return get_r3d50_monai_model(num_classes)
    elif model_name_lower == 'densenet121_monai':
        print("  -> Using MONAI DenseNet-121 model (from scratch).")
        return get_densenet121_monai_model(num_classes)
    elif model_name_lower == 'swinunetr':
        print("  -> Using MONAI SwinUNETR model.")
        return get_swinunetr_model(num_classes)
    elif model_name.lower() == 'unet3d':
        print("  -> Using UNet3D Classifier model.")
        return UNet3DClassifier(in_channels=1, num_classes=num_classes, base_c=32)
    elif model_name.lower() == 'unet3d_resnet_backbone':
        print("  -> Using UNet3D with pre-trained ResNet-18 backbone.")
        return get_unet3d_with_backbone_model(num_classes)
    elif model_name.lower() == 'mil_r3d18':
        print("  -> Using MIL model with MONAI ResNet-18 instance extractor.")
        # Get the base model to use as the instance extractor
        instance_extractor = get_r3d18_monai_model(num_classes)
        # The feature dimension of ResNet-18 is 512
        return MILClassifier(instance_extractor, num_classes, feature_dim=512)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'R3D18_PyTorch', 'R3D50_PyTorch', 'R3D18_MONAI', 'R3D50_MONAI', 'DenseNet121_MONAI', 'SwinUNETR', 'ViT', 'UNet3D', 'UNet3D_ResNet_Backbone', or 'MIL_R3D18'.")