# Quickstart: Aneurysm Experiment Pipeline

## 1. Setup Environment

### Local Development
```bash
# Create Conda environment
conda create -n aneurysm_cnn python=3.11
conda activate aneurysm_cnn

# Install dependencies
pip install -r requirements.txt
```

### HPC Cluster (CUDA/ROCm)
Refer to `environment/cuda/start_script_cuda.slurm` or `environment/rocm/start_script_rocm.slurm` for submitting jobs.

## 2. Generate Datasets (Data Engine)

Run this ONCE to prepare your data. It generates 5 variants (A-E) to allow experimental comparison.

```bash
python data_engine/dataset_gen.py
```
*Input*: defined in `dataset_gen.py` (Default: `/mnt/data/cases-3/nifti`)
*Output*: `/mnt/data/cases-3/dataset_[A-E]_...`

## 3. Configure Experiments

Edit `experiments.json`. Example:

```json
[
  {
    "name": "ResNet18_DatasetA_Base",
    "model": "R3D18_MONAI",
    "data_path_key": "DATASET_A",
    "balancing": "oversampling",
    "augmentation": true,
    "aug_probs": { "flip": 0.5, "noise": 0.1 },
    "hold_out_test_set": true
  }
]
```

## 4. Run Training (Training Engine)

```bash
python training_engine/train_models.py
```

This will:
1. Load `experiments.json`.
2. For each experiment, run K-Fold Cross-Validation.
3. Save artifacts to `experiments/<name>/`.

## 5. Review Results

Check the `experiments/` directory for:
- `summary_metrics.csv`: High-level performance stats.
- `*_roc_curve.png`: Model sensitivity/specificity.
- `*_cm_normalized.png`: Class-wise accuracy.
