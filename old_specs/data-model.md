# Data Model & Contracts

## 1. Experiment Configuration (`experiments.json`)

This JSON file is the single source of truth for the Training Engine. It contains a list of experiment objects.

```json
[
  {
    "name": "string (Required) - Unique identifier for the experiment run",
    "model": "string (Required) - Architecture key (e.g., 'R3D18_MONAI', 'DenseNet121_MONAI')",
    "data_path_key": "string (Required) - Enum: 'DATASET_A'...'DATASET_E'",
    "balancing": "string (Optional) - Enum: 'none', 'oversampling', 'weighted_loss'",
    "augmentation": "boolean (Optional) - Enable on-the-fly transforms",
    "aug_probs": {
      "flip": "float (0.0-1.0)",
      "rotate": "float (0.0-1.0)",
      "noise": "float (0.0-1.0)"
    },
    "hold_out_test_set": "boolean (Optional) - If true, performs final evaluation on reserved split",
    "LEARNING_RATE": "float (Optional)",
    "BATCH_SIZE": "int (Optional)",
    "EPOCHS": "int (Optional)",
    "N_SPLITS": "int (Optional) - Number of CV folds"
  }
]
```

## 2. Artifact Bundle (Output per Experiment)

Each experiment produces a directory `experiments/<name>/`.

| File | Format | Description |
|------|--------|-------------|
| `*_metrics.csv` | CSV | Training history (loss/acc per epoch) + Final F2/Recall/Precision. |
| `*_predictions.csv` | CSV | Sample-level predictions: `filename`, `true_label`, `predicted_label`. |
| `*_cm_raw.png` | PNG | Confusion Matrix (counts). |
| `*_cm_normalized.png` | PNG | Confusion Matrix (percentages). |
| `*_roc_curve.png` | PNG | ROC Curve plot with AUC. |
| `*_model_best.pth` | Binary | PyTorch state dict (Best Validation F2 Score). |
| `*_model_last.pth` | Binary | PyTorch state dict (Final Epoch). |
| `summary_metrics.csv` | CSV | Aggregated stats across all folds (Mean +/- Std Dev). |

## 3. Dataset Registry (Data Engine Output)

The Data Engine produces standard NIfTI files organized by class.

```text
/path/to/dataset_X/
├── 0/ (Negative Class)
│   ├── case123.nii.gz
│   └── ...
└── 1/ (Positive Class - Aneurysm)
    ├── case456.nii.gz
    └── ...
```

**Attributes:**
*   **Format**: NIfTI-1 (`.nii.gz`)
*   **Dtype**: Float32 (normalized)
*   **Dimensions**: 3D (Z, Y, X) - Size depends on variant (A-E) but typically 128x128x128.
