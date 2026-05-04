#!/bin/bash
#SBATCH --job-name=aneurysm_dataset_gen_all
#SBATCH --partition=cpu                 # CPU-only — dataset generation needs no GPU
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4               # Used by 192 run; 176 run is capped at 2
#SBATCH --mem=48G
#SBATCH --time=24:00:00                 # 24 h — runs both shapes sequentially

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

INPUT_DIR=${INPUT_DIR:-$HOME/data/cases-3/nifti}
OUTPUT_BASE=${OUTPUT_BASE:-$HOME/data/cases-3}
VARIANTS=${VARIANTS:-"C D A B E F"}

echo -e "## Job ID:          $SLURM_JOB_ID"
echo -e "## Execution node:  $(hostname -s)"
echo -e "## Input dir:       $INPUT_DIR"
echo -e "## Output base:     $OUTPUT_BASE"
echo -e "## Variants:        $VARIANTS"

## ---------------------------------------------------------------------------
## Environment setup
## ---------------------------------------------------------------------------
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_data_hpc

echo -e "## Active env: $CONDA_DEFAULT_ENV"
echo -e "## Python:     $(which python)"
python -c "import monai, nibabel, numpy; print(f'monai {monai.__version__}  nibabel {nibabel.__version__}  numpy {numpy.__version__}')"
echo ""

cd $HOME/aneurysm_cnn

## ---------------------------------------------------------------------------
## Run 1 — 192×192×128, 4 workers
## ---------------------------------------------------------------------------
echo -e "\n### Pass 1/2 — 192x192x128 | 4 workers ###"

python -u data_engine/dataset_gen.py \
    --input-dir    "$INPUT_DIR"    \
    --output-base  "$OUTPUT_BASE"  \
    --target-shape 192x192x128     \
    --variants     $VARIANTS       \
    --workers      4

STATUS_192=$?
echo -e "### Pass 1/2 finished — exit $STATUS_192 ###"

## ---------------------------------------------------------------------------
## Run 2 — 256×256×176, 2 workers (larger volumes, less RAM per worker)
## ---------------------------------------------------------------------------
echo -e "\n### Pass 2/2 — 256x256x176 | 2 workers ###"

python -u data_engine/dataset_gen.py \
    --input-dir    "$INPUT_DIR"    \
    --output-base  "$OUTPUT_BASE"  \
    --target-shape 256x256x176     \
    --variants     $VARIANTS       \
    --workers      2

STATUS_176=$?
echo -e "### Pass 2/2 finished — exit $STATUS_176 ###"

## ---------------------------------------------------------------------------
## Final status
## ---------------------------------------------------------------------------
echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"

[ $STATUS_192 -eq 0 ] && [ $STATUS_176 -eq 0 ] && exit 0 || exit 1
