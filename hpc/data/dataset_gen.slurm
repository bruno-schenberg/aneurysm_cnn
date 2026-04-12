#!/bin/bash
#SBATCH --job-name=aneurysm_dataset_gen
#SBATCH --partition=cpu                 # CPU-only — dataset generation needs no GPU
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4               # Matches DEFAULT_WORKERS in dataset_gen.py
#SBATCH --mem=32G                       # 256x256x128 float32 volumes peak ~4 GB/worker
#SBATCH --time=12:00:00                 # 12 h wall time for full dataset, all variants

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

## ---------------------------------------------------------------------------
## Configurable parameters (override with env vars at submission time)
##
##   INPUT_DIR=$HOME/data/nifti OUTPUT_DIR=$HOME/data/cases-3 \
##     TARGET_SHAPE=256x256x128 VARIANTS="A B C D" \
##     sbatch hpc/data/dataset_gen.slurm
##
## ---------------------------------------------------------------------------
INPUT_DIR=${INPUT_DIR:-$HOME/data/nifti}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/data/cases-3}
TARGET_SHAPE=${TARGET_SHAPE:-256x256x128}
VARIANTS=${VARIANTS:-B}
N_WORKERS=${N_WORKERS:-4}

## ---------------------------------------------------------------------------
## Job information
## ---------------------------------------------------------------------------
echo -e "## Job ID:          $SLURM_JOB_ID"
echo -e "## Execution node:  $(hostname -s)"
echo -e "## Submit dir:      $SLURM_SUBMIT_DIR"
echo -e "## Input dir:       $INPUT_DIR"
echo -e "## Output dir:      $OUTPUT_DIR"
echo -e "## Target shape:    $TARGET_SHAPE"
echo -e "## Variants:        $VARIANTS"
echo -e "## Workers:         $N_WORKERS \n"

## ---------------------------------------------------------------------------
## Environment setup
## ---------------------------------------------------------------------------
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_data_hpc

## Verify environment
echo -e "## Active env: $CONDA_DEFAULT_ENV"
echo -e "## Python:     $(which python)"
python -c "import monai, nibabel, numpy; print(f'monai {monai.__version__}  nibabel {nibabel.__version__}  numpy {numpy.__version__}')"
echo ""

## ---------------------------------------------------------------------------
## Dataset generation
## ---------------------------------------------------------------------------
cd $HOME/aneurysm_cnn

python -u data_engine/dataset_gen.py \
    --input-dir    "$INPUT_DIR"     \
    --output-dir   "$OUTPUT_DIR"    \
    --target-shape "$TARGET_SHAPE"  \
    --variants     $VARIANTS        \
    --workers      "$N_WORKERS"

STATUS=$?

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"
exit $STATUS
