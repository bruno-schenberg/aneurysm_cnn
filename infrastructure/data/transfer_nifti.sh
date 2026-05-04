#!/usr/bin/env bash
# transfer_nifti.sh
#
# Transfers /mnt/data/nifti to the cluster via scp, then verifies
# file counts match. Skips if already fully transferred.
# Run from your LOCAL machine (not from within SSH).
#
# Usage: bash hpc/transfer_nifti.sh

set -e

CLUSTER="drummond"
LOCAL_DIR="/mnt/data/nifti"
REMOTE_DIR="data/nifti"  # relative to $HOME on the cluster

# SSH options: keepalive prevents broken pipe on long transfers
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=20"

echo "=== Creating remote directory ==="
ssh $SSH_OPTS "$CLUSTER" "mkdir -p ~/$REMOTE_DIR"

echo ""
echo "=== Transferring nifti ==="

local_count=$(find "$LOCAL_DIR" -name "*.nii.gz" | wc -l)

# Check if already fully transferred
remote_count=$(ssh $SSH_OPTS "$CLUSTER" "find ~/$REMOTE_DIR -name '*.nii.gz' 2>/dev/null | wc -l")

if [ "$local_count" -eq "$remote_count" ]; then
    echo "SKIP: nifti already complete ($local_count files)."
    exit 0
fi

echo "Transferring nifti ($local_count files, $remote_count already on cluster)..."
scp $SSH_OPTS -r "$LOCAL_DIR/." "$CLUSTER":~/"$REMOTE_DIR/"

# Verify after transfer
remote_count=$(ssh $SSH_OPTS "$CLUSTER" "find ~/$REMOTE_DIR -name '*.nii.gz' | wc -l")

if [ "$local_count" -eq "$remote_count" ]; then
    echo "OK: nifti — $local_count files verified."
else
    echo "WARNING: nifti — local=$local_count remote=$remote_count — MISMATCH!"
    exit 1
fi

echo ""
echo "=== Transfer complete and verified ==="
