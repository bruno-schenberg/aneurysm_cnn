#!/usr/bin/env bash
# transfer_datasets.sh
#
# Transfers all 5 NIfTI datasets to the cluster via scp, then verifies
# file counts match. Skips datasets already fully transferred.
# Run from your LOCAL machine (not from within SSH).
#
# Usage: bash hpc/transfer_datasets.sh

set -e

CLUSTER="drummond"
LOCAL_DIR="/mnt/data/cases-3"
REMOTE_DIR="data/cases-3"  # relative to $HOME on the cluster

# SSH options: keepalive prevents broken pipe on long transfers
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=20"

DATASETS=(
    "dataset_A_resampled_cropped"
    "dataset_B_resampled_shrunk"
    "dataset_C_cropped"
    "dataset_D_shrunk"
    "dataset_E_isotropic_padded"
)

echo "=== Creating remote directory ==="
ssh $SSH_OPTS "$CLUSTER" "mkdir -p ~/$REMOTE_DIR"

echo ""
echo "=== Transferring datasets ==="
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--- $dataset ---"

    local_count=$(find "$LOCAL_DIR/$dataset" -name "*.nii.gz" | wc -l)

    # Check if already fully transferred
    remote_count=$(ssh $SSH_OPTS "$CLUSTER" "find ~/$REMOTE_DIR/$dataset -name '*.nii.gz' 2>/dev/null | wc -l")

    if [ "$local_count" -eq "$remote_count" ]; then
        echo "SKIP: $dataset already complete ($local_count files)."
        continue
    fi

    echo "Transferring $dataset ($local_count files, $remote_count already on cluster)..."
    scp $SSH_OPTS -r "$LOCAL_DIR/$dataset" "$CLUSTER:~/$REMOTE_DIR/"

    # Verify after transfer
    remote_count=$(ssh $SSH_OPTS "$CLUSTER" "find ~/$REMOTE_DIR/$dataset -name '*.nii.gz' | wc -l")

    if [ "$local_count" -eq "$remote_count" ]; then
        echo "OK: $dataset — $local_count files verified."
    else
        echo "WARNING: $dataset — local=$local_count remote=$remote_count — MISMATCH!"
        exit 1
    fi
done

echo ""
echo "=== All datasets transferred and verified successfully ==="
