"""
Validation check: overfitting test.

Train on a handful of samples with no augmentation for many epochs.
Training loss should drop to near-zero, confirming the loop actually learns.

Run: cd training_engine && python tests/test_overfit.py
"""

import logging

import torch

from dlorchestra.seeding import set_seed
from dlorchestra.splitting import split_data
from dlorchestra.training import build_optimizer, train_one_epoch, validate_one_epoch

from config import DATASET_PATHS, DEFAULT_CONFIG
from src.data import build_dataloaders, get_data_list
from src.models import get_model


def test_overfit() -> None:
    """Train R3D10 on sample_iso_128 with no augmentation — loss must → ~0."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    data_path = DATASET_PATHS["sample_iso_128"]
    if not data_path.exists():
        print(f"SKIP: sample dataset not found at {data_path}")
        return

    seed = 42
    set_seed(seed)

    config = {
        **DEFAULT_CONFIG,
        "name": "_test_overfit",
        "model": "R3D10",
        "data_path_key": "sample_iso_128",
        # No augmentation
        "AUG_FLIP_PROB": 0.0,
        "AUG_ROT90_PROB": 0.0,
        "AUG_SCALE_INTENSITY_PROB": 0.0,
        "AUG_ZOOM_PROB": 0.0,
        "AUG_NOISE_PROB": 0.0,
        # Training params — small and fast
        "EPOCHS": 40,
        "BATCH_SIZE": 2,
        "LEARNING_RATE": 1e-3,
        "WEIGHT_DECAY": 0.0,
        "EARLY_STOPPING_PATIENCE": 0,
        "LR_SCHEDULER": "none",
        "USE_AMP": True,
        "GRAD_ACCUM_STEPS": 1,
        # Data — single fold, no held-out test needed
        "USE_KFOLD": False,
        "TEST_SPLIT_RATIO": 0.15,
        "VAL_SPLIT_RATIO": 0.20,
        "CACHE_RATE": 0.0,
        "NUM_WORKERS": 0,
        "balancing": "none",
        "normalization": "percentile",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    all_data = get_data_list(data_path)
    logger.info("Loaded %d samples", len(all_data))

    # Get the single fold split
    fold_idx, train_files, val_files, test_files = next(
        split_data(
            all_data,
            n_splits=config["N_SPLITS"],
            test_size=config["TEST_SPLIT_RATIO"],
            seed=seed,
            use_kfold=False,
            val_split_ratio=config["VAL_SPLIT_RATIO"],
        )
    )

    logger.info("Train: %d, Val: %d, Test: %d", len(train_files), len(val_files), len(test_files))

    train_loader, val_loader, _ = build_dataloaders(
        train_files, val_files, test_files, config,
    )

    model = get_model("R3D10", num_classes=2).to(device)
    optimizer = build_optimizer(model, config)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and track loss
    use_amp = device == "cuda"
    num_epochs = config["EPOCHS"]
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_accum_steps=1, use_amp=use_amp,
        )
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f",
                epoch, num_epochs, train_loss, train_acc,
            )

    # Evaluate on training set WITHOUT updating weights
    eval_loss, eval_acc, _, _ = validate_one_epoch(
        model, train_loader, criterion, device, use_amp=use_amp,
    )
    logger.info("Eval on train set  loss=%.6f  acc=%.4f", eval_loss, eval_acc)

    loss_threshold = 0.1
    assert eval_loss < loss_threshold, (
        f"Overfitting test FAILED: training-set eval loss {eval_loss:.4f} "
        f"did not drop below {loss_threshold}. The training loop may not be learning."
    )
    assert eval_acc > 0.9, (
        f"Overfitting test FAILED: training-set eval accuracy {eval_acc:.4f} "
        f"is below 0.9."
    )

    print(f"\nOK: Overfitting test passed — eval_loss={eval_loss:.6f}, "
          f"eval_acc={eval_acc:.4f} (threshold: loss<{loss_threshold}, acc>0.9)")


if __name__ == "__main__":
    test_overfit()
