# Research & Technology Decisions

**Feature**: Aneurysm Experiment Pipeline
**Status**: Complete

## Technology Stack Decisions

### 1. Deep Learning Framework
*   **Decision**: **PyTorch** + **MONAI**
*   **Rationale**: 
    *   **PyTorch**: Industry standard for research, dynamic computation graph, vast ecosystem.
    *   **MONAI**: Specialized specifically for medical imaging (3D transforms, sliding window inference, pre-trained medical models). It sits on top of PyTorch, providing domain-specific utilities that would otherwise need to be built from scratch.
*   **Alternatives**: TensorFlow (less flexible for research), Pure PyTorch (would require re-implementing 3D augmentations and I/O logic).

### 2. Image Processing
*   **Decision**: **NiBabel**
*   **Rationale**: The de-facto standard for reading/writing NIfTI files (`.nii.gz`) in Python. Robust header handling.
*   **Alternatives**: SimpleITK (valid choice, but NiBabel is lighter and sufficient for NIfTI).

### 3. Experiment Configuration
*   **Decision**: **JSON** (`experiments.json`)
*   **Rationale**: Human-readable, strict syntax, easily parsed by Python standard library. Allows defining complex nested structures (like augmentation params) clearly.
*   **Alternatives**: YAML (more readable but prone to indentation errors), CLI args (too unwieldy for defining 20+ hyperparameters per experiment).

### 4. Hardware Acceleration
*   **Decision**: **CUDA** (NVIDIA) and **ROCm** (AMD) support.
*   **Rationale**: Maximizes access to available HPC resources.
*   **Implementation**: PyTorch abstracts most of this, but environment isolation (Conda) and SLURM scripts must be specific to the driver version.

## Architecture & Workflow

### Dual-Engine Strategy
The Constitution mandates a split:
1.  **Data Engine**: "Write Once, Read Many". Expensive operations (resampling 500+ volumes) happen once. Output is standard NIfTI on disk.
2.  **Training Engine**: "Iterate Fast". Reads pre-processed data. Focuses on GPU utilization.

### Dataset Strategy (The "5 Variants")
To satisfy User Story 1 (Finding optimal input format), we pre-generate 5 distinct views of the data. This avoids CPU bottlenecks during training (no on-the-fly resampling of massive 512x512x512 volumes).
- **A**: Resampled (1mm) + Cropped
- **B**: Resampled (1mm) + Shrunk
- **C**: Cropped (Native Res)
- **D**: Shrunk (Native Res)
- **E**: Isotropic Resampled + Padded

### Model Strategy
`models.py` uses a Factory Pattern (`get_model`) to instantiate architectures based on string keys in `experiments.json`. This supports rapid switching between `ResNet3D`, `DenseNet`, and `SwinUNETR` without code changes.

## Open Questions Resolved
*   **Model Arch**: Confirmed support for 3D ResNet variants (18/50) as baseline.
*   **Balancing**: Confirmed need for configurable strategies (Weighted Sampling vs Loss).
*   **Test Set**: Confirmed fixed split strategy for consistency.
