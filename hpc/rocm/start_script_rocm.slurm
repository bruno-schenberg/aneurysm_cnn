#!/bin/bash
#SBATCH --job-name=teste_aneurysm_cnn
#SBATCH --partition=gpu-amd
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # CPUs for PyTorch DataLoader
#SBATCH --gpus-per-task=1      # Request 1 GPU
#SBATCH --time=6:00:00

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

## Temporary scratch directory for better I/O performance
SCRATCH=/scratch/local
WRKDIR=$SCRATCH/$SLURM_JOB_ID

## Create scratch directory
srun mkdir -p $WRKDIR

## Copy necessary files to scratch
srun cp -r $HOME/aneurysm_cnn/training_engine $WRKDIR/

## Enter work directory
cd $WRKDIR/training_engine

## Job information
echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Execution Node:                $(hostname -s)"
echo -e "## Submit Directory:              $SLURM_SUBMIT_DIR"
echo -e "## Scratch Directory:             $WRKDIR \n"

#########################################
##-------    Start of Work     ------##
#########################################

## 1. Environment Setup
module purge
module load rocm/6.4.3
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/rocm/rocm-6.4.3/lib:$LD_LIBRARY_PATH

## 2. Architecture Override
## Required for PyTorch to recognise the Instinct MI210 (gfx90a). Format must be decimal.
export HSA_OVERRIDE_GFX_VERSION=9.0.10

## 3. Activate Conda Environment
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_rocm

# 4. Debugging/Verification
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

## 5. Execute Training Script.
## Override experiments file via EXPERIMENTS_FILE env var (optional).
EXPERIMENTS_ARG=""
if [ -n "$EXPERIMENTS_FILE" ]; then
    EXPERIMENTS_ARG="--experiments $EXPERIMENTS_FILE"
fi
python -u train_models.py $EXPERIMENTS_ARG

## 6. Copy the output experiments.
srun cp -r $WRKDIR/training_engine/experiments/. $HOME/aneurysm_cnn/experiments/

## Cleanup Scratch
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"