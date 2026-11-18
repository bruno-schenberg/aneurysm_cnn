#!/usr/bin/env bash
# Overnight: SynthStrip masks (GPU), then the 2D plan (GPU) while the mask
# coverage check + mip_grid_n4mask build run on the CPU. run_2d_local.sh waits
# for that grid before writing the stage-2 configs. Everything is resumable.
#
# Usage (detached; starts at the given time, default now):
#   setsid nohup bash training_engine/scripts/run_overnight.sh 00:00 > ~/overnight.log 2>&1 < /dev/null &
set -uo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -n "${1:-}" ]; then
    start=$(date -d "$1" +%s); [ "$start" -lt "$(date +%s)" ] && start=$(date -d "tomorrow $1" +%s)
    echo "== waiting until $(date -d @"$start")"; sleep $(( start - $(date +%s) ))
fi
echo "== $(date '+%F %T') SynthStrip masks"
bash "$R/data_engine/scripts/run_masks.sh" --masks-only
if [ "$(find /mnt/data/masks_synthstrip -name '*.nii.gz' | wc -l)" -eq 759 ]; then
    echo "== $(date '+%F %T') coverage + mip_grid_n4mask (CPU, background) -> ~/masks_grid.log"
    bash "$R/data_engine/scripts/run_masks.sh" > ~/masks_grid.log 2>&1 &
    sleep 30  # let the coverage check start, so run_2d_local.sh sees it before stage 2
else
    # a background rerun would put SynthStrip back on the GPU next to training
    echo "MASKS INCOMPLETE: 2D runs without the mask arm; rerun run_masks.sh later"
fi
echo "== $(date '+%F %T') 2D plan"
bash "$R/training_engine/scripts/run_2d_local.sh"
wait
echo "== $(date '+%F %T') overnight done"
