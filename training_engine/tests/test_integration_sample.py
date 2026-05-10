import os
import tempfile
import torch
import pytest

from src.data_preprocess import get_data_list, split_data, build_dataloaders
from src.training import run_training_loop, train_one_epoch, validate_one_epoch
from src.plots import save_metrics_to_csv

# We can use a tiny model here to keep it fast, or load a tiny MONAI model.
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten and project directly to 2 classes
        self.fc = torch.nn.Linear(192 * 192 * 192, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # Using a dummy forward to ensure it doesn't OOM and runs quickly
        # We can just pool aggressively:
        return self.fc(x)


class RealTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use an AdaptiveAvgPool3d to reduce spatial dimensions heavily
        self.pool = torch.nn.AdaptiveAvgPool3d((2, 2, 2))
        self.fc = torch.nn.Linear(1 * 2 * 2 * 2, 2)

    def forward(self, x):
        # x is (B, C, D, H, W) e.g., (B, 1, 192, 192, 192)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def test_training_pipeline_with_sample_dataset():
    data_dir = "/mnt/data/nifti-sample-datasets/dataset_A192/"
    
    # 1. Get Data List
    data_list = get_data_list(data_dir)
    assert len(data_list) > 0, "No data found in sample folder"
    
    # Just grab the first 4 samples to keep the test very fast
    data_list = data_list[:4]

    # 2. Split Data
    # For a very small dataset, we just hardcode subsets to avoid empty splits
    train_files = data_list[:2]
    val_files = data_list[2:3]
    test_files = data_list[3:]
    
    # 3. Build DataLoaders
    # use small spatial size to avoid memory issues on CPU testing
    train_loader, val_loader, test_loader = build_dataloaders(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        batch_size=1,
        val_batch_size=1,
        spatial_size=(32, 32, 32), # Resizing for faster test
        cache_rate=0.0, # Disable cache to speed up startup
    )
    
    device = "cpu"
    model = RealTinyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion_cls = torch.nn.CrossEntropyLoss
    
    # 4. Run Training Loop
    metrics_history, total_time, best_model_ckpt = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_cls=criterion_cls,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        class_weights=None,
        patience=0,
        grad_accum_steps=1,
        use_amp=False
    )
    
    assert len(metrics_history) == 2, "Should have 2 epochs in history"
    for m in metrics_history:
        assert "duration_sec" in m, "Missing duration_sec in epoch logs"
        assert m["duration_sec"] > 0, "Duration should be positive"
        assert "val_f2" in m

    # 5. Test metrics CSV writing function (plots.py)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        detailed_results = [
            {"filename": "test1.nii.gz", "true_label": 0, "pred_label": 0},
            {"filename": "test2.nii.gz", "true_label": 1, "pred_label": 1},
        ]
        save_metrics_to_csv(metrics_history, tmp_name, detailed_results)
        
        # Read the CSV back to assert new columns are correctly written
        with open(tmp_name, "r") as f:
            lines = f.readlines()
            header = lines[0].strip().split(",")
            assert "duration_sec" in header, "duration_sec column missing from CSV header"
            
            # Epoch 1 row
            row1 = lines[1].strip().split(",")
            assert float(row1[header.index("duration_sec")]) > 0
            
            # Final Eval row (should be the last line)
            final_row = lines[-1].strip().split(",")
            assert final_row[0] == "final_eval"
            # duration_sec is index 6. In final_eval, it should be empty string
            assert final_row[header.index("duration_sec")] == ""
            
    finally:
        os.remove(tmp_name)
