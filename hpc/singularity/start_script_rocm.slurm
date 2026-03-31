#!/bin/bash
#SBATCH --job-name=aneurysm_cnn
#SBATCH --partition=gpu-amd
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # CPUs for PyTorch DataLoader
#SBATCH --gpus-per-task=1      # Request 1 GPU
#SBATCH --time=6:00:00

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

## Path to the pre-built Singularity image (adjust if stored elsewhere)
SIF=$HOME/aneurysm_cnn_rocm.sif

## Temporary scratch directory for better I/O performance
SCRATCH=/scratch/local
WRKDIR=$SCRATCH/$SLURM_JOB_ID

## Create scratch directory and copy training code
srun mkdir -p $WRKDIR
srun cp -r $HOME/aneurysm_cnn/training_engine $WRKDIR/

## Enter work directory
cd $WRKDIR/training_engine

## Job information
echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Execution Node:                $(hostname -s)"
echo -e "## Submit Directory:              $SLURM_SUBMIT_DIR"
echo -e "## Scratch Directory:             $WRKDIR"
echo -e "## Singularity Image:             $SIF \n"

#########################################
##-------    Start of Work     ------##
#########################################

## Load required modules
module load singularity/3.7.1
module load rocm/6.4.3

## Architecture override for AMD Instinct MI210 (gfx90a).
## Also set inside the container via %environment, but exporting here
## ensures it is inherited even if the container env block is skipped.
export HSA_OVERRIDE_GFX_VERSION=9.0.10

## 1. Verify GPU visibility
echo -e "\n## Verifying GPU & PyTorch inside container... ##"
singularity exec --rocm \
    --bind $WRKDIR:$WRKDIR \
    $SIF python -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm/HIP Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPU(s).")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
'
echo -e "## ...Verification complete. ##\n"

## 2. Run training
## Set EXPERIMENTS_FILE to override which experiments.json to use.
## Default: experiments.json  |  Sample test: experiments_sample.json
EXPERIMENTS_FILE=${EXPERIMENTS_FILE:-experiments.json}
singularity exec --rocm \
    --bind $WRKDIR:$WRKDIR \
    --bind $HOME:$HOME \
    $SIF python -u train_models.py --experiments $EXPERIMENTS_FILE

## 3. Copy results back to home directory
srun cp -r $WRKDIR/training_engine/experiments/. $HOME/aneurysm_cnn/experiments/

## Cleanup scratch
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"
