"""
Diagnostic: augmentation visual check.

Save middle-slice images of a few samples before and after augmentation
to confirm transforms apply and look reasonable.

Run: cd training_engine && python scripts/diagnostics/view_augmentation.py

Output: scripts/diagnostics/augmentation_check/ with before/after PNGs.
"""

from pathlib import Path

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DATASET_PATHS, DEFAULT_CONFIG
from src.data import get_data_list, get_transforms, set_seed


OUTPUT_DIR = Path(__file__).parent / "output" / "augmentation_check"
NUM_SAMPLES = 3
NUM_AUGMENTED = 3  # augmented versions per sample

AXIS_LABELS = ["Sagittal (X)", "Coronal (Y)", "Axial (Z)"]


def _mid_slices(vol: np.ndarray) -> list[np.ndarray]:
    """Extract middle slices along each spatial axis, oriented for display.

    Transpose so Z (superior-inferior) is vertical in sagittal/coronal views.
    """
    mid = [s // 2 for s in vol.shape]
    return [
        np.flipud(vol[mid[0], :, :].T),  # sagittal: (Y,Z)→(Z,Y), flip so superior is up
        np.flipud(vol[:, mid[1], :].T),   # coronal:  (X,Z)→(Z,X), flip so superior is up
        vol[:, :, mid[2]].T,              # axial:    (X,Y)→(Y,X)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    data_path = DATASET_PATHS["sample_iso_128"]
    if not data_path.exists():
        print(f"SKIP: sample dataset not found at {data_path}")
        return

    set_seed(42)
    all_data = get_data_list(data_path)
    samples = all_data[:NUM_SAMPLES]

    config = {**DEFAULT_CONFIG}
    no_aug_transform = get_transforms(config, augment=False)
    aug_transform = get_transforms(config, augment=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(samples):
        exam_id = Path(record["path"]).stem.replace(".nii", "")
        label = record["label"]
        cls_label = "aneurysm" if label == 1 else "control"

        # Original (no augmentation)
        original = no_aug_transform({"image": record["image"]})
        orig_vol = original["image"].numpy().squeeze()

        nrows = 1 + NUM_AUGMENTED
        ncols = 3
        panel_w, panel_h = 4.0, 4.0

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * panel_w, nrows * panel_h),
            squeeze=False,
        )
        fig.patch.set_facecolor("black")
        fig.suptitle(
            f"Sample {i}: {exam_id}  [{cls_label}]",
            fontsize=12, color="white", family="monospace", y=1.005,
        )

        # Column titles
        for j, label_ax in enumerate(AXIS_LABELS):
            axes[0, j].set_title(label_ax, fontsize=9, color="cyan",
                                 family="monospace", pad=8)

        # Row labels
        row_labels = ["Original"] + [f"Augmented {k+1}" for k in range(NUM_AUGMENTED)]

        # Original row
        for j, sl in enumerate(_mid_slices(orig_vol)):
            ax = axes[0, j]
            ax.set_facecolor("black")
            ax.axis("off")
            ax.imshow(sl, cmap="gray", origin="upper", interpolation="bilinear")

        axes[0, 0].text(
            -0.05, 0.5, row_labels[0],
            ha="right", va="center", color="#66ccff", fontsize=9,
            family="monospace", rotation=90,
            transform=axes[0, 0].transAxes,
        )

        # Augmented rows
        for aug_idx in range(NUM_AUGMENTED):
            augmented = aug_transform({"image": record["image"]})
            aug_vol = augmented["image"].numpy().squeeze()
            row = 1 + aug_idx

            for j, sl in enumerate(_mid_slices(aug_vol)):
                ax = axes[row, j]
                ax.set_facecolor("black")
                ax.axis("off")
                ax.imshow(sl, cmap="gray", origin="upper", interpolation="bilinear")

            axes[row, 0].text(
                -0.05, 0.5, row_labels[row],
                ha="right", va="center", color="#66ccff", fontsize=9,
                family="monospace", rotation=90,
                transform=axes[row, 0].transAxes,
            )

        # Shape info overlay on original row
        shape_str = "x".join(str(s) for s in orig_vol.shape)
        axes[0, 0].text(
            0.02, 0.02, shape_str,
            ha="left", va="bottom", color="lime", fontsize=7,
            family="monospace", transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

        plt.tight_layout(pad=0.5)
        out_path = OUTPUT_DIR / f"sample_{i}_{exam_id}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        logger.info("Saved %s", out_path)

    print(f"\nOK: Augmentation visual check — {NUM_SAMPLES} samples saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
