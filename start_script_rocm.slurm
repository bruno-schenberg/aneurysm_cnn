#!/bin/bash
#SBATCH --job-name=teste_aneurysm_cnn
#SBATCH --partition=gpu-amd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # CPUs for PyTorch DataLoader
#SBATCH --gpus-per-task=1      # Request 1 GPU
#SBATCH --time=01:00:00

echo -e "\n## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

## Temporary scratch directory for better I/O performance
SCRATCH=/scratch/local
WRKDIR=$SCRATCH/$SLURM_JOB_ID

## Job information
echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Execution Node:                $(hostname -s)"
echo -e "## Submit Directory:              $SLURM_SUBMIT_DIR"
echo -e "## Scratch Directory:             $WRKDIR \n"

## Create scratch directory
srun mkdir -p $WRKDIR

## Copy necessary files to scratch
## Assuming 'train_models.py' is in 'training_engine'
srun cp -r $HOME/aneurysm_cnn/training_engine $WRKDIR/

## Enter work directory
cd $WRKDIR/training_engine

#########################################
##-------    Start of Work     ------##
#########################################

## 1. Environment Setup
## We will NOT load the system's rocm module. The PyTorch Conda environment is
## self-contained and should not conflict with system-level ROCm libraries.
module purge

## 2. OPTIONAL: Architecture Override
## Uncomment the line below ONLY if you get "hipErrorNoBinaryForGpu" errors.
## This forces PyTorch to treat the GPU as a specific architecture (common fix for MI200 series, including MI210).
export HSA_OVERRIDE_GFX_VERSION=10.3.0

## 3. Activate Conda Environment
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_rocm  # <--- Updated environment name

## 4. Debugging/Verification
echo -e "\n## Verifying Environment & GPU... ##"
echo "Active Conda Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"

## Check user group permissions (The root cause of the issue)
echo -e "\n## Checking User and Group Permissions:"
echo -n "User Groups: "; groups
echo "Device Permissions:"
ls -l /dev/kfd
ls -l /dev/dri/card*

## Check if the HIP runtime in Conda can see the GPU (bypasses PyTorch)
echo -e "\n## rocminfo Output:"
rocminfo || echo "rocminfo command failed. This indicates a problem with the Conda environment's ROCm stack."

## Check for missing shared libraries in PyTorch
echo -e "\n## Checking PyTorch library dependencies (ldd):"
ldd $(python -c "import torch; print(torch.__path__[0])")/lib/libtorch_hip.so

## Check if ROCm is visible to PyTorch
python -c "import os, torch; print(f'PyTorch Version: {torch.__version__}'); print(f'HSA_OVERRIDE_GFX_VERSION: {os.environ.get(\"HSA_OVERRIDE_GFX_VERSION\")}') ; print(f'ROCm/HIP Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

## Check system-level GPU status (The AMD equivalent of nvidia-smi)
echo -e "\n## rocm-smi Output:"
## Use the absolute path to rocm-smi since we did not load the module
/opt/rocm/bin/rocm-smi
echo -e "## ...Verification Complete. ##\n"

## 5. Execute Training Script.
## Run the python script directly. The environment is now clean and fully
## controlled by Conda, which should resolve the detection issue.
python -u train_models.py

## Cleanup Scratch
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"