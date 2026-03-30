#!/bin/bash
#SBATCH --job-name=test_singularity_rocm
#SBATCH --partition=gpu-amd
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=0:30:00

echo -e "## Job started at $(date +'%d-%m-%Y as %T') #####################\n"

## Path to the pre-built Singularity image (adjust if stored elsewhere)
SIF=$HOME/aneurysm_cnn_rocm.sif

## Temporary scratch directory
SCRATCH=/scratch/local
WRKDIR=$SCRATCH/$SLURM_JOB_ID

srun mkdir -p $WRKDIR
srun cp -r $SLURM_SUBMIT_DIR/../rocm/test_rocm.py $WRKDIR/

cd $WRKDIR

## Job information
echo -e "## Job ID:                        $SLURM_JOB_ID"
echo -e "## Execution Node:                $(hostname -s)"
echo -e "## Submit Directory:              $SLURM_SUBMIT_DIR"
echo -e "## Scratch Directory:             $WRKDIR"
echo -e "## Singularity Image:             $SIF \n"

#########################################
##-------    Start of Work     ------##
#########################################

## Load Singularity module (as required by the Drummond cluster)
module load singularity/3.7.1

export HSA_OVERRIDE_GFX_VERSION=9.0.a

## 1. rocm-smi from the host (outside container, for hardware diagnostics)
echo -e "\n## Running rocm-smi (host) for diagnostics... ##"
srun rocm-smi
echo -e "## ...rocm-smi complete. ##"

## 2. Relevant environment variables
echo -e "\n## Relevant GPU environment variables: ##"
env | grep -E 'ROCR_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|HSA_OVERRIDE_GFX_VERSION' \
    || echo " -> None found."

## 3. Run the Python test inside the container
echo -e "\n## Running Python test inside Singularity container... ##"
srun singularity exec --rocm \
    --bind $WRKDIR:$WRKDIR \
    $SIF python -u test_rocm.py
echo -e "## ...container test complete. ##\n"

## Cleanup
rm -rf $WRKDIR

echo -e "\n## Job finished at $(date +'%d-%m-%Y as %T') ###################"
