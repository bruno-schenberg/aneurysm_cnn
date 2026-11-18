#!/usr/bin/env bash
# run_masks.sh — 2D-1b mask arm (tasks 2.3b): SynthStrip brain masks for our
# 759 exams -> coverage check (picks the mask dilation) -> N4 + masked MIP grid.
#
# SynthStrip weights + script come from the freesurfer/synthstrip Docker image
# (extracted to ~/.cache/aneurysm/synthstrip without a Docker daemon). Runs in its
# own venv there: surfa 0.6.3 (2024) breaks on NumPy >= 2.4 and recent SciPy, and
# Python 3.14 has no wheels old enough, so the venv is Python 3.12:
#   uv venv ~/.cache/aneurysm/synthstrip/venv --python 3.12
#   VIRTUAL_ENV=... uv pip install torch==2.13.0 "numpy<2" "scipy<1.14" surfa==0.6.3 \
#       --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
# Resumable.
#
# Usage: bash data_engine/scripts/run_masks.sh [--masks-only]
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
SS="$HOME/.cache/aneurysm/synthstrip/freesurfer"
MASKS=/mnt/data/masks_synthstrip
D=/mnt/data/datasets
for f in /mnt/data/nifti/*/*.nii.gz; do
    out="$MASKS/$(basename "$(dirname "$f")")/$(basename "$f")"
    [ -s "$out" ] && continue
    mkdir -p "$(dirname "$out")"
    "$SS/../venv/bin/python" "$SS/mri_synthstrip" -i "$f" -m "$out" -g --model "$SS/models/synthstrip.1.pt" > /dev/null
done
echo "masks: $(find $MASKS -name '*.nii.gz' | wc -l)/759"
[ "${1:-}" = "--masks-only" ] && exit 0  # GPU part only (run_overnight.sh)
cd "$R/data_engine"
.venv/bin/python scripts/check_mask_coverage.py --masks "$MASKS" | tee "$MASKS/coverage.txt"
DIL=$(sed -n 's/.*smallest dilation keeping 100%: \([0-9]*\) mm/\1/p' "$MASKS/coverage.txt")
[ -n "$DIL" ] || { echo "No safe dilation; skipping the n4mask grid (report as a finding)"; exit 0; }
.venv/bin/python -m steps.step_mip.generate_mip_grid --source /mnt/data/nifti --output-root "$D/mip_grid_n4mask" \
    --n4 --mask-dir "$MASKS" --mask-dilate-mm "$DIL" --aspects stretch --workers 8
