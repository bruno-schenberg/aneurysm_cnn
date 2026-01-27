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

## 1. LOAD SYSTEM MODULES (Crucial for ROCm)
## This must happen before activating Conda
module purge
module load rocm/6.0.1

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

## Check if ROCm is visible to PyTorch
python -c "import os, torch; print(f'PyTorch Version: {torch.__version__}'); print(f'HSA_OVERRIDE_GFX_VERSION: {os.environ.get(\"HSA_OVERRIDE_GFX_VERSION\")}') ; print(f'ROCm/HIP Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

## Check system-level GPU status (The AMD equivalent of nvidia-smi)
echo -e "\n## rocm-smi Output:"
rocm-smi
echo -e "## ...Verification Complete. ##\n"

## 5. Execute Training Script.
## Unsetting LD_LIBRARY_PATH forces PyTorch to use its own bundled ROCm libraries,
## avoiding conflicts with the system's `rocm/6.0.1` module.
## We run python directly without `srun` to ensure a clean environment.
echo "Unsetting LD_LIBRARY_PATH to prioritize PyTorch's ROCm libraries..."
unset LD_LIBRARY_PATH
python -u train_models.py

## Cleanup Scratch
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"