#!/usr/bin/env bash
# setup.sh — Install nnU-Net and download pre-trained aneurysm models.
# Run ON the cluster after activating the conda env.
#
# Usage:
#   conda activate aneurysm_cnn_cuda
#   bash ~/cnn_for_aneurysm/segmentation/setup.sh

set -euo pipefail

SEG_DIR="$HOME/cnn_for_aneurysm/segmentation"
MODEL_DIR="$SEG_DIR/models"
ZENODO_RECORD="13386859"

echo "=== Installing nnU-Net v2 ==="
# Pinned: 2.8.1 produced the 2026-09-12 segmentation results.
# On ROCm, install with --no-deps — its torch!=2.9.* pin would swap the ROCm torch for a PyPI CUDA wheel.
pip install nnunetv2==2.8.1

echo ""
echo "=== Downloading multi-centric aneurysm model from Zenodo ($ZENODO_RECORD) ==="
mkdir -p "$MODEL_DIR/zenodo_download"
cd "$MODEL_DIR/zenodo_download"

# Get download URLs from Zenodo API
echo "Fetching file list..."
URLS=$(python -c "
import urllib.request, json
data = json.load(urllib.request.urlopen('https://zenodo.org/api/records/$ZENODO_RECORD'))
for f in data['files']:
    print(f['links']['self'], f['key'], f['size'])
")

echo "Files to download:"
echo "$URLS"
echo ""

# Download each file
while IFS=' ' read -r url filename size; do
    if [ -f "$filename" ]; then
        local_size=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename" 2>/dev/null)
        if [ "$local_size" = "$size" ]; then
            echo "Already downloaded: $filename ($size bytes)"
            continue
        fi
    fi
    echo "Downloading: $filename ($size bytes)..."
    # Zenodo drops slow connections mid-transfer — resume until the size matches.
    for attempt in $(seq 1 50); do
        curl -L -C - -o "$filename" "$url" || true
        [ "$(stat -c%s "$filename" 2>/dev/null)" = "$size" ] && break
        echo "  incomplete (attempt $attempt), resuming in 30s..."
        sleep 30
    done
    [ "$(stat -c%s "$filename")" = "$size" ] || { echo "ERROR: $filename download failed"; exit 1; }
done <<< "$URLS"

echo ""
echo "=== Extracting model files ==="
for f in *.zip; do
    [ -f "$f" ] && { echo "Extracting $f..."; unzip -o "$f" -d "${f%.zip}" || true; }
done
for f in *.tar.gz *.tgz; do
    [ -f "$f" ] && { echo "Extracting $f..."; tar xzf "$f" || true; }
done

echo ""
echo "=== Downloaded contents ==="
find . -maxdepth 3 -type f | head -50
echo ""

# Try to find nnU-Net model structure
echo "=== Looking for nnU-Net model structure ==="
DATASET_JSON=$(find . -name "dataset.json" -type f 2>/dev/null | head -1)
if [ -n "$DATASET_JSON" ]; then
    # nnU-Net stores dataset.json in the trainer dir; nnUNet_results needs the Dataset<ID>_* dir above it
    MODEL_ROOT=$(dirname "$(dirname "$DATASET_JSON")")
    echo "Found model at: $MODEL_ROOT"
    echo "Contents:"
    find "$MODEL_ROOT" -maxdepth 2 -type f | head -20

    # Symlink into nnUNet_results structure
    RESULTS_DIR="$MODEL_DIR/nnUNet_results"
    mkdir -p "$RESULTS_DIR"
    DATASET_DIR_NAME=$(basename "$MODEL_ROOT")
    if [ ! -e "$RESULTS_DIR/$DATASET_DIR_NAME" ]; then
        ln -s "$(realpath "$MODEL_ROOT")" "$RESULTS_DIR/$DATASET_DIR_NAME"
        echo ""
        echo "Symlinked model to: $RESULTS_DIR/$DATASET_DIR_NAME"
    fi
else
    echo "WARNING: No dataset.json found. Model structure may differ from standard nnU-Net."
    echo "Inspect the downloaded files manually and set up nnUNet_results accordingly."
fi

echo ""
echo "=== Cloning connectomicslab/Aneurysm_Detection (backup model) ==="
CLAB_DIR="$MODEL_DIR/connectomicslab"
if [ -d "$CLAB_DIR" ]; then
    echo "Already cloned."
else
    git clone --depth 1 https://github.com/connectomicslab/Aneurysm_Detection.git "$CLAB_DIR"
    echo "Cloned to $CLAB_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "nnU-Net model dir: $MODEL_DIR/nnUNet_results"
echo "Connectomicslab dir: $CLAB_DIR"
echo ""
echo "Next: run inference with"
echo "  bash segmentation/run_inference.sh [source_data_dir]"
