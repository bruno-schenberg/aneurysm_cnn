#!/bin/bash
#SBATCH --job-name=dataset_gen_128
#SBATCH --partition=cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

INPUT_DIR=${INPUT_DIR:-$HOME/data/cases-3/nifti}
OUTPUT_BASE=${OUTPUT_BASE:-$HOME/data/cases-3}

echo -e "## Job ID:          $SLURM_JOB_ID"
echo -e "## Execution node:  $(hostname -s)"
echo -e "## Input dir:       $INPUT_DIR"
echo -e "## Output base:     $OUTPUT_BASE"

source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_data_hpc

echo -e "## Active env: $CONDA_DEFAULT_ENV"
echo -e "## Python:     $(which python)"

cd $HOME/aneurysm_cnn

python -u data_engine/dataset_gen.py \
    --input-dir   "$INPUT_DIR"   \
    --output-base "$OUTPUT_BASE" \
    --target-shape 128x128x128   \
    --variants C D A B E F       \
    --workers 4

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"
