#!/bin/bash
#SBATCH --job-name=diagnose_rocm
#SBATCH --partition=gpu-amd
#SBATCH --output=diagnose_rocm_%j.log
#SBATCH --error=diagnose_rocm_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=0:10:00

echo "============================================================"
echo " ROCm Diagnostic Report — for HPC support"
echo " Date:    $(date)"
echo " Node:    $(hostname -s)"
echo " Job ID:  $SLURM_JOB_ID"
echo " User:    $USER"
echo "============================================================"

# ----------------------------------------------------------------
# 1. Modules
# ----------------------------------------------------------------
echo ""
echo "--- Loaded modules at job start ---"
module list 2>&1

echo ""
echo "--- rocm/6.4.3 module definition ---"
module show rocm/6.4.3 2>&1

# ----------------------------------------------------------------
# 2. ROCm installations on disk
# ----------------------------------------------------------------
echo ""
echo "--- ROCm directories in /opt ---"
ls -la /opt/rocm* -d 2>&1

echo ""
echo "--- /opt/rocm symlink resolves to ---"
readlink -f /opt/rocm 2>&1

echo ""
echo "--- rocm/6.4.3 target directories (expected by module) ---"
for dir in /opt/rocm-6.4.3/bin /opt/rocm-6.4.3/lib /opt/rocm-6.4.3/include; do
    if [ -d "$dir" ]; then
        echo "  EXISTS   $dir"
    else
        echo "  MISSING  $dir"
    fi
done

# ----------------------------------------------------------------
# 3. Device access
# ----------------------------------------------------------------
echo ""
echo "--- GPU device files ---"
ls -la /dev/kfd /dev/dri/renderD* 2>&1

echo ""
echo "--- User groups ---"
groups

echo ""
echo "--- KFD kernel driver version ---"
cat /sys/module/amdkfd/version 2>/dev/null || echo "(not available at /sys/module/amdkfd/version)"

# ----------------------------------------------------------------
# 4. rocm-smi (uses sysfs — works without HSA)
# ----------------------------------------------------------------
echo ""
echo "--- rocm-smi (host, outside any container) ---"
module load rocm/6.4.3 2>&1
rocm-smi 2>&1

# ----------------------------------------------------------------
# 5. rocminfo with rocm/6.4.3 (expected version)
# ----------------------------------------------------------------
echo ""
echo "--- rocminfo with module rocm/6.4.3 ---"
echo "ROCM_PATH=$ROCM_PATH"
export HSA_OVERRIDE_GFX_VERSION=9.0.10
rocminfo 2>&1

# ----------------------------------------------------------------
# 6. rocminfo with rocm/6.0.1 (only version actually installed)
# ----------------------------------------------------------------
echo ""
echo "--- rocminfo with module rocm/6.0.1 ---"
module purge 2>&1
module load rocm/6.0.1 2>&1
echo "ROCM_PATH=$ROCM_PATH"
rocminfo 2>&1

# ----------------------------------------------------------------
# 7. Python diagnostics
# ----------------------------------------------------------------
echo ""
echo "--- Python diagnostics ---"
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/.conda/etc/profile.d/conda.sh" 2>/dev/null \
    || echo "WARNING: could not source conda — skipping Python section"

conda activate aneurysm_cnn_rocm 2>&1 && \
    python "$HOME/aneurysm_cnn/hpc/rocm/diagnose_rocm.py" 2>&1 \
    || echo "conda env not available — skipping Python section"

echo ""
echo "============================================================"
echo " End of diagnostic report"
echo " Send diagnose_rocm_${SLURM_JOB_ID}.log to HPC support"
echo "============================================================"
