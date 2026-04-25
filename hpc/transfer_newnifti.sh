#!/usr/bin/env bash
# transfer_newnifti.sh
#
# Transfers the newly converted NIfTI exams from /mnt/data/newnifti into
# the existing nifti folder on the cluster without re-transferring all exams.
# Preserves the class subdirectory structure (0/ and 1/).
# Run from your LOCAL machine (not from within SSH).
#
# Usage: bash hpc/transfer_newnifti.sh

set -e

CLUSTER="drummond"
LOCAL_DIR="/mnt/data/newnifti"
REMOTE_DIR="data/nifti"  # merges into the existing nifti folder on the cluster

SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=20"

local_total=$(find "$LOCAL_DIR" -name "*.nii.gz" | wc -l)
echo "Local files to transfer: $local_total"

echo ""
echo "=== Creating remote subdirectories ==="
ssh $SSH_OPTS "$CLUSTER" "mkdir -p ~/$REMOTE_DIR/0 ~/$REMOTE_DIR/1"

echo ""
echo "=== Transferring ==="
for class_dir in 0 1; do
    [ -d "$LOCAL_DIR/$class_dir" ] || continue
    n=$(find "$LOCAL_DIR/$class_dir" -name "*.nii.gz" | wc -l)
    [ "$n" -eq 0 ] && continue
    echo "  class $class_dir — $n file(s)..."
    scp $SSH_OPTS "$LOCAL_DIR/$class_dir/"*.nii.gz "$CLUSTER:~/$REMOTE_DIR/$class_dir/"
done

echo ""
echo "=== Verifying ==="
# Build a check command that counts how many of our specific files exist remotely
check_cmd=""
while IFS= read -r f; do
    class_dir=$(basename "$(dirname "$f")")
    name=$(basename "$f")
    check_cmd+="[ -f ~/$REMOTE_DIR/$class_dir/$name ] && echo ok; "
done < <(find "$LOCAL_DIR" -name "*.nii.gz")

remote_ok=$(ssh $SSH_OPTS "$CLUSTER" "$check_cmd" | wc -l)

if [ "$remote_ok" -eq "$local_total" ]; then
    echo "OK: all $local_total file(s) verified on cluster."
else
    echo "WARNING: only $remote_ok/$local_total found on cluster."
    exit 1
fi

echo ""
echo "=== Transfer complete ==="
