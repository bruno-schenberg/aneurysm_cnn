#!/usr/bin/env bash
# run_inference.sh — Run nnU-Net inference on our dataset and evaluate.
# Runs ON the cluster. Expects setup.sh to have been run first.
#
# Usage:
#   bash segmentation/run_inference.sh                    # uses c90z_384 (default)
#   bash segmentation/run_inference.sh ~/data/nifti       # uses original NIfTIs
#   bash segmentation/run_inference.sh ~/data/datasets/iso_256

set -euo pipefail

cd "$HOME/cnn_for_aneurysm"

# Activate conda
for conda_sh in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    [ -f "$conda_sh/etc/profile.d/conda.sh" ] && { source "$conda_sh/etc/profile.d/conda.sh"; break; }
done
conda activate "${CONDA_ENV:-aneurysm_cnn_cuda}"

# --- Configuration ---
SOURCE_DIR="${1:-$HOME/data/datasets/c90z_384}"
DATASET_CSV="$HOME/cnn_for_aneurysm/dataset/dataset.csv"

SEG_DIR="$HOME/cnn_for_aneurysm/segmentation"
INPUT_DIR="$SEG_DIR/nnunet_input"
OUTPUT_DIR="$SEG_DIR/nnunet_output"
EVAL_DIR="$SEG_DIR/evaluation"

export nnUNet_raw="$SEG_DIR/nnUNet_raw"
export nnUNet_preprocessed="$SEG_DIR/nnUNet_preprocessed"
export nnUNet_results="$SEG_DIR/models/nnUNet_results"

echo "## nnU-Net Inference Pipeline"
echo "## $(date)"
echo "## Source: $SOURCE_DIR"
echo ""

# --- Verify model exists ---
if [ ! -d "$nnUNet_results" ]; then
    echo "ERROR: nnU-Net results dir not found at $nnUNet_results"
    echo "Run setup.sh first."
    exit 1
fi

# --- Step 1: Prepare input ---
echo "=== Step 1: Prepare input ==="
rm -rf "$INPUT_DIR"
python "$SEG_DIR/prepare_input.py" "$SOURCE_DIR" "$INPUT_DIR" --dataset-csv "$DATASET_CSV"

N_FILES=$(find "$INPUT_DIR" -name "*_0000.nii.gz" | wc -l)
echo "Input files: $N_FILES"
if [ "$N_FILES" -eq 0 ]; then
    echo "ERROR: No input files prepared. Check source directory: $SOURCE_DIR"
    exit 1
fi
echo ""

# --- Step 2: Model parameters ---
echo "=== Step 2: Model config ==="

# Multi-centric nnU-Net from Zenodo 13386859
DATASET_ID=610
CONFIG="3d_fullres"
FOLDS="0 1 2 3 4"

echo "Dataset: $DATASET_ID, Config: $CONFIG, Folds: $FOLDS"
echo ""

# --- Step 3: Run nnU-Net predict ---
echo "=== Step 3: nnU-Net inference ==="
echo "This may take a while for $N_FILES volumes..."
echo ""

mkdir -p "$OUTPUT_DIR"

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -f $FOLDS \
    --disable_tta \
    --verbose

echo ""
echo "Predictions saved to $OUTPUT_DIR"
echo ""

# --- Step 4: Evaluate ---
echo "=== Step 4: Evaluate ==="
python "$SEG_DIR/evaluate.py" "$OUTPUT_DIR" "$SEG_DIR/mapping.csv" --output-dir "$EVAL_DIR"

echo ""
echo "## Done at $(date)"
