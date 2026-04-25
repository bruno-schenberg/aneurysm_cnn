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

echo "=== Creating remote subdirectories ==="
ssh $SSH_OPTS "$CLUSTER" "mkdir -p ~/$REMOTE_DIR/0 ~/$REMOTE_DIR/1"

echo ""
echo "=== Transferring new NIfTI files ==="

for class_dir in 0 1; do
    local_files=("$LOCAL_DIR/$class_dir/"*.nii.gz)

    # Skip if no files in this class dir
    [ -e "${local_files[0]}" ] || continue

    local_count=$(find "$LOCAL_DIR/$class_dir" -name "*.nii.gz" | wc -l)
    echo ""
    echo "--- class $class_dir ($local_count file(s)) ---"

    for f in "$LOCAL_DIR/$class_dir/"*.nii.gz; do
        name=$(basename "$f")
        remote_exists=$(ssh $SSH_OPTS "$CLUSTER" "[ -f ~/$REMOTE_DIR/$class_dir/$name ] && echo yes || echo no")
        if [ "$remote_exists" = "yes" ]; then
            echo "  SKIP: $name already on cluster."
            continue
        fi
        echo "  Transferring $name..."
        scp $SSH_OPTS "$f" "$CLUSTER:~/$REMOTE_DIR/$class_dir/$name"
        echo "  OK: $name"
    done
done

echo ""
echo "=== Verifying ==="
local_total=$(find "$LOCAL_DIR" -name "*.nii.gz" | wc -l)
verified=0
for f in $(find "$LOCAL_DIR" -name "*.nii.gz"); do
    class_dir=$(basename "$(dirname "$f")")
    name=$(basename "$f")
    remote_exists=$(ssh $SSH_OPTS "$CLUSTER" "[ -f ~/$REMOTE_DIR/$class_dir/$name ] && echo yes || echo no")
    if [ "$remote_exists" = "yes" ]; then
        verified=$((verified + 1))
    else
        echo "  MISSING on cluster: $class_dir/$name"
    fi
done

if [ "$verified" -eq "$local_total" ]; then
    echo "OK: all $local_total new file(s) verified on cluster."
else
    echo "WARNING: $verified/$local_total verified — check missing files above."
    exit 1
fi

echo ""
echo "=== Transfer complete ==="
