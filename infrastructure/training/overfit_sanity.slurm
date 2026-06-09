#!/bin/bash
#SBATCH --job-name=overfit_sanity
#SBATCH --partition=gpu-amd
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

## Temporary scratch directory for better I/O performance
SCRATCH=/scratch/local
WRKDIR=$SCRATCH/$SLURM_JOB_ID

## Create scratch directory and copy training engine
srun mkdir -p $WRKDIR
srun cp -r $HOME/aneurysm_cnn/training_engine $WRKDIR/

cd $WRKDIR/training_engine

echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Execution Node:                $(hostname -s)"
echo -e "## Submit Directory:              $SLURM_SUBMIT_DIR"
echo -e "## Scratch Directory:             $WRKDIR \n"

## 1. Environment Setup
module purge
module load rocm/6.4.3
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/rocm/rocm-6.4.3/lib:$LD_LIBRARY_PATH

## 2. Architecture Override
export HSA_OVERRIDE_GFX_VERSION=9.0.10

## 3. Dataset root and backup dir
export DATA_ROOT=$HOME/data/cases-3
export EXPERIMENT_BACKUP_DIR=$HOME/aneurysm_cnn/experiments_overfit

## 4. Activate environment
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_rocm

## 5. Verify GPU
echo -e "\n## Verifying Environment & GPU... ##"
echo "Active Conda Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"
python -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm/HIP Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPUs.")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
'
echo -e "## ...Verification Complete. ##\n"

## 6. Run overfit sanity experiments
python -u train_models.py --experiments experiments_overfit.json

## 7. Copy results back
if [ -d "$WRKDIR/training_engine/experiments_overfit_results" ]; then
    mkdir -p $HOME/aneurysm_cnn/experiments_overfit
    srun cp -r $WRKDIR/training_engine/experiments_overfit_results/. $HOME/aneurysm_cnn/experiments_overfit/
else
    echo "WARNING: experiments_overfit_results directory not found — training may have failed."
fi

## Cleanup Scratch
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"
