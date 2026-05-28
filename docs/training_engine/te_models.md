## models.py

`training_engine/src/models.py` is the model registry for the aneurysm CNN training engine. It defines every supported architecture and exposes a single factory entry point, `get_model`, that constructs any of them by name.

---

### Module-level constant: `SUPPORTED_MODELS`

A plain `list[str]` that enumerates every valid architecture key accepted by `get_model`. It exists to (a) document what is available, (b) provide a human-readable error message on invalid names, and (c) act as a cross-reference in `_strip_classifier`. It is **not** used programmatically as a dispatch table — the actual dispatch is an `if/elif` chain in `get_model`.

---

### Model families implemented

| Family | Members | Pretraining |
|---|---|---|
| 3D ResNet (MONAI / MedicalNet) | `R3D10`, `R3D18_MONAI`, `R3D50` | MedicalNet (23 medical datasets), 400-class head |
| 3D ResNet (torchvision) | `R3D18` | Kinetics-400 video classification |
| Vision Transformer | `UNETR` | None (from scratch) |
| Swin Transformer | `SwinUNETR` | BTCV segmentation (swinViT encoder only) |
| Dense connections | `DenseNet121` | None (from scratch) |
| Custom UNet | `UNet3D`, `UNet3DWithBackbone` | None / MedicalNet encoder |
| Multiple Instance Learning | `MIL_R3D18` | MedicalNet encoder |

All models share the same external interface: input `(B, 1, D, H, W)`, output logits `(B, num_classes)`.

---

### Functions

---

#### `get_r3d18_pytorch_model`

**What it does:** Loads the torchvision R3D-18 video backbone and adapts it for single-channel 3D medical image classification.

**Design rationale:** R3D-18 was pretrained on Kinetics-400 (RGB video), so its first convolution expects 3 input channels. Rather than discarding those learned filters, the RGB weights are averaged along the channel dimension (`mean(dim=1, keepdim=True)`) to produce equivalent 1-channel filters — preserving spatial structure while matching the medical input format. This approach transfers more weight information than random re-initialisation.

**Parameters:**
- `num_classes` (`int`): Number of output classes.
- `weights` (`Optional[str]`, default `"default"`): If `"default"`, loads `R3D_18_Weights.DEFAULT` (Kinetics-400 pretrained); any other value loads without pretrained weights.

**Returns:** `torch.nn.Module` — the modified torchvision R3D-18 model.

**Hardcoded hyperparameters:**
- Input channels forced to 1 (single-channel MRI/CTA).
- Channel averaging uses `dim=1` — matches the RGB channel axis of the Conv3d weight tensor `(out_c, in_c, D, H, W)`.

---

#### `get_r3d18_monai_model`

**What it does:** Loads (or falls back to random-init) a MONAI ResNet-18 with MedicalNet pretraining and replaces its 400-class head with a task-specific head.

**Design rationale:** MedicalNet weights were trained across 23 medical image segmentation datasets, making them a stronger starting point than Kinetics-400 video weights for medical imaging tasks. The 400-class head is an artefact of MedicalNet's multi-task pretraining and is always replaced.

**Parameters:**
- `num_classes` (`int`): Number of output classes.

**Returns:** `torch.nn.Module` — MONAI ResNet-18 with replaced classification head.

**Hardcoded hyperparameters:**
- `pretrained=True` first (falls back silently on any exception).
- `num_classes=400` on construction to match the pretrained checkpoint; overwritten immediately after.
- `n_input_channels=1`, `spatial_dims=3` — fixed for 3D single-channel volumes.
- ResNet-18 encoder output dimensionality: 512 (implicit, from `model.fc.in_features`).

---

#### `get_r3d10_model`

**What it does:** Loads (or falls back to random-init) a MONAI ResNet-10 with MedicalNet pretraining and replaces its head.

**Design rationale:** ResNet-10 is the lightest MONAI ResNet variant, useful when compute or memory is limited. Same pretraining strategy as ResNet-18.

**Parameters:**
- `num_classes` (`int`): Number of output classes.

**Returns:** `torch.nn.Module` — MONAI ResNet-10 with replaced classification head.

**Hardcoded hyperparameters:** Same as `get_r3d18_monai_model`; encoder output also 512-dimensional (noted in the comment, matching ResNet-18 for this family).

---

#### `get_r3d50_model`

**What it does:** Loads (or falls back to random-init) a MONAI ResNet-50 with MedicalNet pretraining and replaces its head.

**Design rationale:** ResNet-50 uses bottleneck blocks giving higher capacity at the cost of more parameters and memory.

**Parameters:**
- `num_classes` (`int`): Number of output classes.

**Returns:** `torch.nn.Module` — MONAI ResNet-50 with replaced classification head.

**Hardcoded hyperparameters:** Same as other MONAI ResNets; encoder output 2048-dimensional (bottleneck architecture).

---

#### `get_densenet121_monai_model`

**What it does:** Constructs a MONAI DenseNet-121 for 3D classification, trained from scratch.

**Design rationale:** DenseNet's dense connectivity (each layer receives feature maps from all preceding layers) encourages feature reuse and gradient flow, which can be beneficial on small medical datasets where training signal is scarce.

**Parameters:**
- `num_classes` (`int`): Number of output classes.

**Returns:** `torch.nn.Module` — MONAI DenseNet-121.

**Hardcoded hyperparameters:**
- `init_features=64`, `growth_rate=32`, `block_config=(6, 12, 24, 16)` — these are the canonical DenseNet-121 architecture parameters (Huang et al., 2017). They are fixed constants that define the DenseNet-121 variant specifically, not tunable defaults.
- `spatial_dims=3`, `in_channels=1` — fixed for the task.

---

#### `get_swinunetr_model`

**What it does:** Convenience wrapper that constructs a `SwinUNETRClassifier` with fixed defaults.

**Parameters:**
- `num_classes` (`int`): Number of output classes.

**Returns:** `torch.nn.Module` — `SwinUNETRClassifier` instance.

**Hardcoded hyperparameters:**
- `img_size=(128, 128, 128)` — passed for API compatibility; not forwarded to the MONAI SwinUNETR constructor (see class note).
- `in_channels=1`, `use_pretrained=True`.

---

#### `get_unetr_model`

**What it does:** Convenience wrapper that constructs a `UNETRClassifier` with ViT-Base hyperparameters.

**Parameters:**
- `num_classes` (`int`): Number of output classes.
- `img_size` (`tuple`, default `(128, 128, 128)`): Spatial size of input volumes; forwarded to the ViT for positional embeddings.

**Returns:** `torch.nn.Module` — `UNETRClassifier` instance.

**Hardcoded hyperparameters:**
- `hidden_size=768`, `mlp_dim=3072`, `num_heads=12` — the canonical ViT-Base configuration (Dosovitskiy et al., 2020). The `mlp_dim=3072` is exactly `4 × hidden_size`, the standard FFN expansion ratio.

---

#### `get_unet3d_with_backbone_model`

**What it does:** Convenience wrapper that constructs a `UNet3DWithBackbone` with `base_c=32`.

**Parameters:**
- `num_classes` (`int`): Number of output classes.

**Returns:** `torch.nn.Module` — `UNet3DWithBackbone` instance.

**Hardcoded hyperparameters:**
- `base_c=32` — decoder base channel count; hardcoded here rather than being a parameter of the factory function.

---

#### `_strip_classifier`

**What it does:** Replaces the final classification linear layer of a model with `nn.Identity()` in-place, and returns the feature dimensionality that was stripped.

**Design rationale:** Used by `get_model` to prepare any image backbone for multimodal late fusion. By operating in-place and returning the dimension, callers do not need to know the per-architecture attribute path for the classifier. All architecture-specific knowledge is centralised here.

**Parameters:**
- `model` (`nn.Module`): An already-constructed image model.
- `model_name` (`str`): Architecture name (case-insensitive), used to select the correct attribute path.

**Returns:** `int` — feature dimensionality of the stripped layer.

**Raises:** `ValueError` if `model_name` is not a recognised architecture.

**Hardcoded hyperparameters / complex control flow — flag:** Contains a 4-branch `if/elif/elif/else` that encodes the per-architecture attribute path to the classifier (`model.fc` for ResNets, `model.class_layers.out` for DenseNet, `model.classifier` for the rest). This must be updated every time a new model is added, creating a maintenance coupling between `SUPPORTED_MODELS`, `get_model`, and `_strip_classifier`. The three locations are not automatically kept in sync.

---

#### `get_model` (factory)

**What it does:** Constructs and returns any supported model by name, optionally wrapping it in `MultiModalWrapper` for multimodal fusion.

**Parameters:**
- `model_name` (`str`): Architecture key, case-insensitive; must be in `SUPPORTED_MODELS`.
- `num_classes` (`int`): Number of output classes.
- `use_tabular` (`bool`, default `False`): If `True`, strips the classifier and wraps in `MultiModalWrapper`.
- `tabular_dim` (`int`, default `2`): Dimensionality of the tabular feature vector (age + gender by default).
- `spatial_size` (`tuple`, default `(128, 128, 128)`): Forwarded to `get_unetr_model`; not used by any other architecture.

**Returns:** `torch.nn.Module` — configured model ready for training.

**Raises:** `ValueError` for unknown `model_name`.

**Complex control flow — flag:** The dispatch is a 10-branch `if/elif` chain (one branch per architecture). This could be replaced with a registry dictionary mapping name strings to constructor callables, eliminating the chain entirely. A dict-based approach would also make adding a new model a single-site change (one dict entry) rather than requiring edits to `get_model`, `SUPPORTED_MODELS`, and `_strip_classifier` separately.

---

### Classes

---

#### `ConvBlock3D`

**What it does:** Implements the canonical UNet double-convolution building block — two sequential `Conv3d → BatchNorm3d → ReLU` layers — as a reusable module.

**Design rationale:** Two 3×3×3 convolutions give a 5×5×5 effective receptive field with fewer parameters than a single 5×5×5 kernel and an extra non-linearity between them. BatchNorm after each convolution (before ReLU) follows the ResNet convention for training stability. `ReLU(inplace=True)` reduces memory overhead.

**Parameters (`__init__`):**
- `in_channels` (`int`): Input channel count.
- `out_channels` (`int`): Output channel count (same for both convolutions).

**`forward` parameters:**
- `x` (`torch.Tensor`): Feature map of shape `(B, in_channels, D, H, W)`.

**Returns (`forward`):** `torch.Tensor` of shape `(B, out_channels, D, H, W)`.

**Hardcoded hyperparameters:**
- `kernel_size=3`, `padding=1` — preserves spatial dimensions (same-convolution).

---

#### `UNet3DClassifier`

**What it does:** Adapts the classic 3D UNet (encoder + skip connections + decoder) for scan-level classification by replacing the segmentation output with Global Average Pooling and a linear head.

**Design rationale:** Skip connections force the network to maintain multi-scale feature information, potentially useful for detecting small structures (aneurysms) that need both local detail and global vascular context. GAP is used instead of flattening because flattening the final `128³` feature map would be computationally infeasible.

**Parameters (`__init__`):**
- `in_channels` (`int`, default `1`): Input channels.
- `num_classes` (`int`, default `2`): Output classes.
- `base_c` (`int`, default `32`): Base channel count; each encoder level doubles this.

**`forward` parameters:**
- `x` (`torch.Tensor`): Input volume `(B, in_channels, D, H, W)`.

**Returns (`forward`):** `torch.Tensor` of logits `(B, num_classes)`.

**Hardcoded hyperparameters:**
- `base_c=32` (when called from `get_model`; the class itself accepts it as a parameter).
- 4 encoder stages + bottleneck, giving 5 spatial scales. Spatial resolution: `128→64→32→16→8`. Channel counts: `base_c, 2×, 4×, 8×, 16×` (i.e., 32, 64, 128, 256, 512 with `base_c=32`).
- `MaxPool3d(2)` for downsampling, `ConvTranspose3d(kernel_size=2, stride=2)` for upsampling — standard UNet choices.
- `AdaptiveAvgPool3d((1,1,1))` — reduces any spatial size to a scalar per channel.

---

#### `SwinUNETRClassifier`

**What it does:** Repurposes the SwinUNETR segmentation architecture as a 3D volumetric classifier by discarding the CNN decoder and attaching a Global Average Pooling + linear head to the deepest Swin Transformer encoder output.

**Design rationale:** BTCV-pretrained SwinUNETR weights provide a strong medical-image feature initialisation. Only the encoder (`swinViT`) is loaded from the checkpoint because the decoder is discarded. The deepest encoder stage (`x4_out`) has the richest semantic content at the coarsest spatial resolution. `use_checkpoint=True` during construction enables gradient checkpointing to reduce memory usage at the cost of recomputation.

**Parameters (`__init__`):**
- `img_size` (`tuple`): Accepted for API compatibility; not forwarded to the MONAI `SwinUNETR` constructor (MONAI >= 1.4 removed this argument).
- `in_channels` (`int`): Number of input channels.
- `num_classes` (`int`): Number of output classes.
- `feature_size` (`int`, default `48`): Base feature dimension of the Swin Transformer. Must match the pretrained checkpoint (checkpoint uses 48).
- `use_pretrained` (`bool`, default `True`): If `True`, attempts to load encoder weights from a hardcoded local path.

**`forward` parameters:**
- `x` (`torch.Tensor`): Input volume `(B, in_channels, D, H, W)`.

**Returns (`forward`):** `torch.Tensor` of logits `(B, num_classes)`.

**Hardcoded hyperparameters:**
- `out_channels=14` — matches the BTCV 14-class segmentation checkpoint; arbitrary since the decoder is unused.
- Pretrained weights path: `/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth` — absolute, environment-specific, and fragile.
- Classifier input dimension: `feature_size * 16` (768 for `feature_size=48`), which is the SwinViT deepest stage channel count — a fixed architectural ratio.
- `weights_only=True` in `torch.load` — security-safe loading.

**Flag:** The hardcoded weight path `/root/.cache/monai/models/...` assumes a `root` home directory and will silently fall back to random init on any other system, making the pretrained-vs-scratch distinction non-obvious at runtime.

---

#### `UNETRClassifier`

**What it does:** Repurposes the UNETR pure ViT encoder for 3D classification by replacing the CNN decoder with mean pooling over patch tokens and a linear head.

**Design rationale:** UNETR's ViT encoder attends globally across all 512 patch tokens from the first layer, unlike SwinUNETR's windowed local attention. This captures long-range anatomical relationships. No pretrained checkpoint is loaded because mapping MONAI UNETR weight keys across versions is brittle; this model is used to test the global-attention architectural hypothesis. Mean pooling treats all patches equally, unlike attention pooling, but is simple and avoids introducing extra learned parameters.

**Parameters (`__init__`):**
- `num_classes` (`int`): Number of output classes.
- `img_size` (`tuple`, default `(128, 128, 128)`): Must match actual input for positional embeddings.
- `hidden_size` (`int`, default `768`): Transformer embedding dimension (ViT-Base).
- `mlp_dim` (`int`, default `3072`): FFN hidden dimension (ViT-Base, = 4 × hidden_size).
- `num_heads` (`int`, default `12`): Attention heads (ViT-Base); must divide `hidden_size`.

**`forward` parameters:**
- `x` (`torch.Tensor`): Input volume `(B, 1, D, H, W)`.

**Returns (`forward`):** `torch.Tensor` of logits `(B, num_classes)`.

**Hardcoded hyperparameters:**
- `in_channels=1`, `out_channels=14`, `feature_size=16` — decoder output channels are irrelevant (decoder unused); `feature_size=16` is UNETR's patch projection dimension.
- `proj_type="conv"`, `norm_name="instance"`, `res_block=True` — standard UNETR construction choices.
- For a `128³` input with patch size 16: N_patches = `(128/16)³ = 512` tokens, pooled over `dim=1`.

---

#### `UNet3DWithBackbone`

**What it does:** Combines a pretrained MONAI ResNet-18 encoder (from MedicalNet) with a UNet-style decoder and skip connections for multi-scale classification.

**Design rationale:** A plain fine-tuned ResNet uses only the deepest features. Adding a UNet decoder with skip connections reintroduces spatial detail from earlier encoder stages, which may help discriminate small aneurysms that require fine boundary information alongside coarse semantic context.

**Parameters (`__init__`):**
- `num_classes` (`int`, default `2`): Number of output classes.
- `base_c` (`int`, default `32`): Base channel count for the final decoder stage.

**`forward` parameters:**
- `x` (`torch.Tensor`): Input volume `(B, 1, D, H, W)`.

**Returns (`forward`):** `torch.Tensor` of logits `(B, num_classes)`.

**Hardcoded hyperparameters:**
- Decoder channel widths are hardcoded to match ResNet-18's layer output sizes: 512 (layer4/bottleneck), 256 (layer3), 128 (layer2), 64 (layer1/stem).
- `up0` output channels = `base_c`; `dec0` input = `base_c + 64` (to accommodate the 64-channel stem skip).
- `base_c=32` when called from `get_unet3d_with_backbone_model`.

**Flag (fragile internal API access):** The `forward` method accesses MONAI ResNet-18's internals by name: `backbone.conv1`, `backbone.bn1`, `backbone.act`, `backbone.maxpool`, `backbone.layer1`–`backbone.layer4`. The use of `.act` (not `.relu`) is MONAI-specific and differs from torchvision. Any MONAI internal refactor would break this silently at runtime rather than at import time.

---

#### `MILClassifier`

**What it does:** Implements attention-based Multiple Instance Learning (MIL) for 3D volumetric classification, treating a scan as a bag of patches and learning which patches are diagnostically relevant.

**Design rationale:** Aneurysms occupy a small sub-region of the full volume. MIL allows the model to learn patch-level discriminative features without per-patch labels, providing implicit spatial attention. The attention mechanism uses Tanh (not ReLU) to model both excitatory and inhibitory evidence across patches. Softmax normalisation ensures attention weights sum to 1, making the bag aggregation a convex combination of patch features.

**Parameters (`__init__`):**
- `instance_extractor` (`nn.Module`): A pretrained 3D CNN whose final linear layer will be stripped via `children()[:-1]`.
- `num_classes` (`int`): Number of output classes.
- `feature_dim` (`int`): Dimensionality of the per-patch feature vector after head removal (512 for ResNet-18).

**`forward` parameters:**
- `x` (`torch.Tensor`): Either 5D `(B, C, D, H, W)` (single instance) or 6D `(B, N, C, D, H, W)` (bag of N patches).

**Returns (`forward`):** `torch.Tensor` of logits `(1, num_classes)`.

**Hardcoded hyperparameters:**
- Attention MLP hidden dimension: `128` (Linear(feature_dim → 128) → Tanh → Linear(128 → 1)).
- `feature_dim=512` when called from `get_model` — hardcoded to match ResNet-18's encoder output.

**Flags:**

1. **Batch size = 1 constraint (silent failure):** The `forward` method squeezes the batch dimension only when `x.shape[0] == 1`. Passing batch size > 1 will leave a 6D tensor that the instance extractor cannot process, causing a shape error at runtime rather than a clear assertion. There is no guard or error message.

2. **Fragile head stripping:** `nn.Sequential(*list(instance_extractor.children())[:-1])` relies on `children()` returning a flat, ordered list that matches the forward pass. For MONAI ResNet-18 this happens to work, but it fails for any architecture with a non-trivial module hierarchy (e.g., sub-modules that contain further sub-modules not in the top-level child list).

---

#### `MultiModalWrapper`

**What it does:** Late-fusion multimodal wrapper that concatenates image backbone features with a small tabular MLP embedding (age, gender) and feeds the result through a fusion head for joint classification.

**Design rationale:** Late fusion (feature concatenation after independent processing of each modality) is simpler than early or intermediate fusion and less prone to overfitting on only 2 tabular features. The tabular branch is intentionally shallow (2 linear layers) to prevent it from dominating the learning signal. Dropout (0.5) in the fusion head regularises the combined representation.

**Parameters (`__init__`):**
- `image_model` (`nn.Module`): Image backbone with classifier already stripped.
- `image_feature_dim` (`int`): Dimensionality of image features.
- `tabular_dim` (`int`, default `2`): Number of tabular features (age + gender).
- `num_classes` (`int`, default `2`): Number of output classes.

**`forward` parameters:**
- `image` (`torch.Tensor`): Volumetric input `(B, 1, D, H, W)`.
- `tabular` (`torch.Tensor`): Tabular input `(B, tabular_dim)`.

**Returns (`forward`):** `torch.Tensor` of logits `(B, num_classes)`.

**Hardcoded hyperparameters:**
- Tabular branch: `Linear(tabular_dim → 16) → ReLU → Linear(16 → 16)` — 16-dim embedding, fixed regardless of `tabular_dim`.
- Fusion head: `Linear(image_feature_dim + 16 → 128) → ReLU → Dropout(0.5) → Linear(128 → num_classes)`.
- Dropout rate `0.5` — a standard regularisation default, not tuned.

---

### Adding a new model

Adding a new model currently requires changes in **three separate places**, which are not automatically enforced:

1. **`SUPPORTED_MODELS`** — add the new name string.
2. **`get_model`** — add an `elif` branch that calls the new constructor.
3. **`_strip_classifier`** — add an `elif` branch mapping the new name to its classifier attribute path (required only if `use_tabular=True` is ever used with the new model).

There is no single-file registration pattern; the three locations are coupled by convention only.

---

### Complexity flags summary

| Location | Issue | Suggested simplification |
|---|---|---|
| `get_model` | 10-branch `if/elif` dispatch over model name strings | Replace with a `dict[str, Callable]` registry mapping names to constructor lambdas; dispatch becomes `registry[name](...)` |
| `_strip_classifier` | 4-branch `if/elif` encoding per-architecture classifier attribute paths | Merge into the same registry dict as an additional key per entry (e.g., a `strip_fn` field), eliminating the separate function |
| `SUPPORTED_MODELS` | Manually maintained list not tied to the actual dispatch table | Auto-derive from the registry dict keys |
| `MILClassifier.forward` | Silent failure when batch size > 1 | Add `assert x.shape[0] == 1` with an explanatory message |
| `MILClassifier.__init__` | Head stripping via `children()[:-1]` is order-dependent and fragile | Use named attribute access (e.g., `model.fc = nn.Identity()`) or a dedicated `build_instance_extractor` function |
| `UNet3DWithBackbone.forward` | Direct access to MONAI ResNet internals (`backbone.act`, `backbone.bn1`, etc.) | Wrap in a dedicated `ResNetFeatureExtractor` class that exposes skip-connection outputs via a clean interface |
| `SwinUNETRClassifier.__init__` | Hardcoded absolute weight path `/root/.cache/monai/models/...` | Accept as a constructor parameter or read from config/environment variable |

---

