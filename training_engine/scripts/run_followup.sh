#!/usr/bin/env bash
# D10.8 follow-ups: 2D-3_512 seeds 1+2, then late fusion
# on N4 MIPs at 384 and 512 (x3 seeds). Configs are hand-written in configs/final/.
# Resumable: finished experiments (summary.json) are skipped.
#
# Usage (detached, survives the terminal; optional start time):
#   setsid nohup bash training_engine/scripts/run_followup.sh 00:00 > ~/followup.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export DATASETS_ROOT=${DATASETS_ROOT:-/mnt/data/datasets}
PY=.venv/bin/python
if [ -n "${1:-}" ]; then
    start=$(date -d "$1" +%s); [ "$start" -lt "$(date +%s)" ] && start=$(date -d "tomorrow $1" +%s)
    echo "== waiting until $(date -d @"$start")"; sleep $(( start - $(date +%s) ))
fi
train() {
    for f in "$@"; do
        echo "== $(date '+%F %T') $f"
        $PY -u train_models.py --experiments "$f" || echo "FAILED $f"
    done
}

# before the readiness check, which flags a referenced dataset that is not built;
# always called, since it skips finished exams (a killed run leaves a partial folder)
echo "== $(date '+%F %T') generating mip_grid_n4/vessel90_stretch_512"
(cd ../data_engine && .venv/bin/python -m steps.step_mip.generate_mip_grid --source /mnt/data/nifti \
    --output-root "$DATASETS_ROOT/mip_grid_n4" --windows vessel90 --aspects stretch --sizes 512 \
    --workers 8 --n4) || exit 1
$PY scripts/check_datasets.py --quick || exit 1
train configs/final/2D-3_512_s1.json configs/final/2D-3_512_s2.json configs/final/2D-8_n4_latefusion_*.json
echo "== $(date '+%F %T') followup done"
