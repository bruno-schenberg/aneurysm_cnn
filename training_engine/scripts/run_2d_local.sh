#!/usr/bin/env bash
# The whole 2D plan on the local GPU, stage by stage: 2D-1 (x3 seeds) -> 2D-1b + 2D-2 (x3 seeds) ->
# 768 MIPs for the winner -> 2D-3..2D-6.
# Re-runnable: finished experiments (summary.json) are skipped, and the next
# stage's configs are only written once every run of the previous one finished.
#
# Usage (detached, survives the terminal):
#   setsid nohup bash training_engine/scripts/run_2d_local.sh > ~/run_2d.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export DATASETS_ROOT=${DATASETS_ROOT:-/mnt/data/datasets}
PY=.venv/bin/python
MASKS=/mnt/data/masks_synthstrip
train() {
    for f in "$@"; do
        [ -f "$f" ] || continue
        echo "== $(date '+%F %T') $f"
        $PY -u train_models.py --experiments "$f" || echo "FAILED $f"
    done
}

$PY scripts/check_datasets.py --quick || exit 1
train configs/final/2D-1_*.json
# a MIP grid still being written (run_overnight.sh builds mip_grid_n4mask meanwhile)
# would give stage 2 a half-built dataset
while pgrep -f "steps.step_mip.generate_mip_grid|check_mask_coverage.py" > /dev/null; do sleep 60; done
$PY scripts/make_final_configs.py --stage 2 || exit 1
train configs/final/2D-1b_*.json configs/final/2D-2_*.json
$PY scripts/make_final_configs.py --stage 3 || exit 1

# 2D-3 768: generate only the winner's window/aspect, in the winner's family
key=$($PY -c "import json; print(json.load(open('configs/final/2D-3_768.json'))[0]['data_path_key'])")
fam=${key%%/*}; var=${key#*/}; win=${var%_*_*}; aspect=${var#"$win"_}; aspect=${aspect%_*}
if [ ! -d "$DATASETS_ROOT/$key" ]; then
    case $fam in
        mip_grid) extra=() ;;
        mip_grid_n4) extra=(--n4) ;;
        mip_grid_n4mask) extra=(--n4 --mask-dir "$MASKS" --mask-dilate-mm
            "$(sed -n 's/.*smallest dilation keeping 100%: \([0-9]*\) mm/\1/p' "$MASKS/coverage.txt")") ;;
        *) echo "no 768 recipe for $fam"; exit 1 ;;
    esac
    echo "== $(date '+%F %T') generating $key"
    (cd ../data_engine && .venv/bin/python -m steps.step_mip.generate_mip_grid --source /mnt/data/nifti \
        --output-root "$DATASETS_ROOT/$fam" --windows "$win" --aspects "$aspect" --sizes 768 \
        --workers 8 "${extra[@]}") || exit 1
fi
train configs/final/2D-3_*.json configs/final/2D-4_*.json configs/final/2D-5_*.json configs/final/2D-6_*.json
echo "== $(date '+%F %T') 2D done"
