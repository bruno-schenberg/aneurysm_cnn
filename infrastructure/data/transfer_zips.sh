#!/usr/bin/env bash
# transfer_zips.sh
#
# Rsyncs all .zip archives from LOCAL_DIR to the cluster, then verifies
# the file count matches. Runs over a single SSH connection — no repeated
# password prompts. Partial transfers are safe to resume (rsync skips
# already-transferred files automatically).
#
# Usage: bash infrastructure/data/transfer_zips.sh

set -e

CLUSTER="drummond"
LOCAL_DIR="/mnt/data/raw"
REMOTE_DIR="data/raw_zips"   # relative to $HOME on the cluster

# SSH options: keepalive prevents broken pipe on long transfers
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=20"

echo "=== Creating remote directory ==="
ssh $SSH_OPTS "$CLUSTER" "mkdir -p ~/$REMOTE_DIR"

echo ""
echo "=== Transferring zip archives ==="

local_count=$(find "$LOCAL_DIR" -maxdepth 1 -name "*.zip" | wc -l)
echo "Local zips found: $local_count"

if [ "$local_count" -eq 0 ]; then
    echo "ERROR: No .zip files found in $LOCAL_DIR"
    exit 1
fi

# Check how many are already on the cluster so we can report progress.
remote_count=$(ssh $SSH_OPTS "$CLUSTER" "find ~/$REMOTE_DIR -maxdepth 1 -name '*.zip' 2>/dev/null | wc -l")
echo "Already on cluster: $remote_count"

if [ "$local_count" -eq "$remote_count" ]; then
    echo "SKIP: all $local_count zip(s) already present on cluster."
    exit 0
fi

echo "Transferring ($local_count total, $remote_count already done)..."

# -a  : archive (preserves timestamps, permissions)
# -P  : show per-file progress + keep partial files for resume
# -e  : pass SSH options through rsync's ssh tunnel (single connection)
# --include / --exclude : transfer only .zip files, nothing else
rsync -aP \
    -e "ssh $SSH_OPTS" \
    --include="*.zip" \
    --exclude="*" \
    "$LOCAL_DIR/" \
    "$CLUSTER":~/"$REMOTE_DIR/"

echo ""
echo "=== Verifying transfer ==="
remote_count=$(ssh $SSH_OPTS "$CLUSTER" "find ~/$REMOTE_DIR -maxdepth 1 -name '*.zip' | wc -l")

if [ "$local_count" -eq "$remote_count" ]; then
    echo "OK: $local_count zip(s) verified on cluster."
else
    echo "WARNING: local=$local_count  remote=$remote_count — MISMATCH!"
    exit 1
fi

echo ""
echo "=== Next steps ==="
echo "  1. SSH into the cluster:  ssh $CLUSTER"
echo "  2. Pull latest code:      cd ~/aneurysm_cnn && git pull"
echo "  3. Run the inventory scan:"
echo "       conda activate aneurysm_cnn_data_hpc"
echo "       python data_engine/inventory_scan.py \\"
echo "           --zip-dir    ~/data/raw_zips \\"
echo "           --extract-dir ~/data/raw_extract \\"
echo "           --log-dir    data_engine/output/inventory"
echo ""
echo "=== Transfer complete ==="
