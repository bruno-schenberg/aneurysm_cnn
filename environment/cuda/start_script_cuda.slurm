#!/bin/bash
#SBATCH --job-name=teste_aneurysm_cnn
#SBATCH --partition=gpu-nvidia
#SBATCH --output=environment/cuda/logs/%x_%j.out
#SBATCH --error=environment/cuda/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # CPUs para o DataLoader do PyTorch
#SBATCH --gpus-per-task=1      # Solicita 1 GPU para a tarefa
#SBATCH --time=01:00:00

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
## We will load the system's cuda module.
module purge
module load cuda/12.1

## 3. Activate Conda Environment
source $HOME/miniconda3/bin/activate
conda activate aneurysm_cnn_cuda

# 4. Debugging/Verification
echo -e "\n## Verifying Environment & GPU... ##"
echo "Active Conda Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"
python -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPUs.")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
'
echo -e "## ...Verification Complete. ##\n"

## 5. Execute Training Script.
## Run the python script directly. The environment is now clean and fully
## controlled by Conda, which should resolve the detection issue.
python -u train_models.py

## 6. Copy the output experiments.
srun cp -r $WRKDIR/training_engine/experiments $HOME/aneurysm_cnn/training_engine 

## Cleanup Scratch
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"