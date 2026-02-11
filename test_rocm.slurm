#!/bin/bash
#SBATCH --job-name=teste_sbatch
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

## Setup trap to remove scratch directory on exit
cleanup() {
    echo -e "\n## Cleaning up scratch directory: $WRKDIR ##"
    rm -rf "$WRKDIR"
    echo "## Cleanup complete. ##"
}
trap cleanup EXIT

## Create scratch directory
srun mkdir -p $WRKDIR

## Copy necessary files to scratch
srun cp -r $SLURM_SUBMIT_DIR/test_rocm.py $WRKDIR/

## Enter work directory
cd $WRKDIR

## Job information
echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Execution Node:                $(hostname -s)"
echo -e "## Submit Directory:              $SLURM_SUBMIT_DIR"
echo -e "## Scratch Directory:             $WRKDIR \n"

#########################################
##-------    Start of Work     ------##
#########################################

## 1. Environment Setup
## The PyTorch ROCm build is self-contained and includes the necessary libraries.
## Loading a system-level ROCm module can cause library conflicts.
## We will rely on the libraries bundled with PyTorch in the Conda environment.
# module purge
# module load rocm/6.0.1

## 2. Architecture Override
## This is required for PyTorch to work correctly with the Instinct MI210 (gfx90a) GPU.
export HSA_OVERRIDE_GFX_VERSION=9.0.a

## 3. Activate Conda Environment
# The following is the recommended way to initialize conda in a script
eval "$(conda shell.bash hook)"
conda activate aneurysm_cnn_rocm

## 4. Diagnostic: Check if ROCm can see the GPU
echo -e "\n## Running rocm-smi for diagnostics... ##"
# We use the full path since we are not loading the rocm module
srun /opt/rocm/bin/rocm-smi
echo -e "## ...rocm-smi complete. ##"

# 5. Debugging/Verification
echo -e "\n## Verifying Environment & GPU... ##"
echo "## Relevant environment variables from batch script:"
env | grep -E 'ROCR_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|HSA_OVERRIDE_GFX_VERSION' || echo " -> No relevant ROCm/GPU environment variables found in batch shell."
echo "Active Conda Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"

echo -e "\n## Running Python verification with srun... ##"
srun --export=ALL python -c '
import torch
import os
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm/HIP Available: {torch.cuda.is_available()}")
print(f"  - ROCR_VISIBLE_DEVICES: {os.environ.get('ROCR_VISIBLE_DEVICES')}")
print(f"  - HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
print(f"  - LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPUs.")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
'
echo -e "## ...srun verification complete. ##\n"

## 6. Execute Training Script.
srun --export=ALL python -u test_rocm.py

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"