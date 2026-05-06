# Models TODO & Research Notes

## 1. Current Models Lacking Pre-Trained Weights
As discovered during the refactoring process, the following models are currently initialized from scratch (random weights), making them prone to overfitting on small aneurysm datasets:

*   **`DenseNet121`**: Currently uses MONAI's DenseNet121 trained from scratch. 
*   **`UNETR`**: Pure ViT encoder + CNN decoder, currently initialized randomly.
*   **`UNet3D`**: A custom architecture, so naturally no pretrained weights exist.
*   **`SwinUNETR`**: Technically supports pre-trained BTCV weights, but currently hardcodes a local cache path (`/root/.cache/monai/models/swin_unetr.base_5000ep_f48_fe.pth`). If this file isn't manually downloaded by the user, it silently falls back to training from scratch.

### Action Items for Current Models:
- [ ] **Fix SwinUNETR**: Update the code to automatically download the BTCV weights via the Hugging Face Hub or `monai.bundle` instead of relying on a hardcoded local file.
- [ ] **Find DenseNet121 3D Weights**: Look into MedicalNet or Med3D to see if a 3D DenseNet checkpoint exists that we can load into the MONAI architecture.

---

## 2. Recommended Models to Explore (MONAI Zoo & Literature)

For aneurysm detection (which involves identifying very small, localized anomalies in large 3D MRA/CTA volumes), researchers generally favor architectures that preserve high-resolution features and leverage self-supervised medical pre-training.

Here are a few highly relevant models available in MONAI or popular in the literature that we should consider adding:

### A. HighResNet (Available in MONAI)
*   **Why:** Traditional ResNets and UNets downsample the image heavily (e.g., dividing spatial resolution by 32). For aneurysms (which can be just a few voxels wide), heavy downsampling destroys the signal. HighResNet maintains high-resolution feature maps throughout the network using dilated convolutions.
*   **Status:** Native MONAI support (`monai.networks.nets.HighResNet`). 

### B. EfficientNet-3D (Available in MONAI)
*   **Why:** EfficientNet uses a compound scaling method that balances depth, width, and resolution. It often outperforms ResNet variants with significantly fewer parameters. This is excellent for 3D medical imaging where GPU memory is a bottleneck.
*   **Status:** Native MONAI support (`monai.networks.nets.EfficientNet3D`). Pre-trained weights for 3D are sometimes available through third-party MedicalNet-style repositories.

### C. SegResNet Encoder (Available in MONAI / MONAI Model Zoo)
*   **Why:** SegResNet is consistently the top-performing architecture in medical imaging competitions (like BraTS). While it's a segmentation network, we can easily strip the decoder and use the encoder + average pooling for classification (just like we did for SwinUNETR).
*   **Pre-training:** MONAI's Model Zoo has extensive pre-trained bundles for SegResNet.

### D. Self-Supervised Learning (SSL) Models (MONAI Zoo)
*   **Why:** Recent literature shows that Self-Supervised pre-training (like masked volume modeling) on unlabeled CT/MRI scans yields much better representations than transferring from video datasets (Kinetics) or heavily supervised datasets.
*   **Status:** MONAI hosts several SSL pre-trained ViTs and SwinUNETRs in their Model Zoo. Switching to an SSL pre-trained SwinUNETR encoder instead of the BTCV (supervised) one might yield a major performance boost.

### E. 3D UX-Net / ConvNeXt-3D (Literature)
*   **Why:** Transformers (like UNETR) are memory-heavy for large 3D volumes. UX-Net and ConvNeXt-3D are modern convolutional architectures that simulate Transformer behavior (using large kernel sizes like 7x7x7) but remain lightweight. 
*   **Status:** Highly popular in 2023/2024 medical imaging papers. Would require importing from external GitHub repos as they aren't native to MONAI yet.