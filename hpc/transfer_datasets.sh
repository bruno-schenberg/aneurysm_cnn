#!/usr/bin/env bash
# transfer_datasets.sh
#
# Transfers all 5 NIfTI datasets to the cluster via scp, then verifies
# file counts match. Run from your LOCAL machine (not from within SSH).
#
# Usage: bash hpc/transfer_datasets.sh

set -e

CLUSTER="drummond"
LOCAL_DIR="/mnt/data/cases-3"
REMOTE_DIR="data/cases-3"  # relative to $HOME on the cluster

DATASETS=(
    "dataset_A_resampled_cropped"
    "dataset_B_resampled_shrunk"
    "dataset_C_cropped"
    "dataset_D_shrunk"
    "dataset_E_isotropic_padded"
)

echo "=== Creating remote directory ==="
ssh "$CLUSTER" "mkdir -p ~/$REMOTE_DIR"

echo ""
echo "=== Transferring datasets ==="
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--- $dataset ---"
    scp -r "$LOCAL_DIR/$dataset" "$CLUSTER:~/$REMOTE_DIR/"

    local_count=$(find "$LOCAL_DIR/$dataset" -name "*.nii.gz" | wc -l)
    remote_count=$(ssh "$CLUSTER" "find ~/$REMOTE_DIR/$dataset -name '*.nii.gz' | wc -l")

    if [ "$local_count" -eq "$remote_count" ]; then
        echo "OK: $dataset — $local_count files transferred and verified."
    else
        echo "WARNING: $dataset — local=$local_count remote=$remote_count — MISMATCH!"
        exit 1
    fi
done

echo ""
echo "=== All datasets transferred and verified successfully ==="
