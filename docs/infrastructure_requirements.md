# Infrastructure & Cluster Requirements

## 5. Infrastructure

### 5.1 Environment Management

- Local (`aneurysm_cnn_data`) and HPC (`aneurysm_cnn_data_hpc`) conda environments must pin Python at the same granularity (patch-level `3.11.15`, not minor-only `3.11`).
- `pydicom`, `nilearn`, and `itk` are present locally but absent on HPC; if any data-engine script importing these is expected to run on HPC, the HPC env must be updated to include them.
- `pytest` and `ruff` are intentionally local-only; their absence on HPC is acceptable and need not be reconciled.
- The `setuptools` pin comment must state the same reason in both YAMLs (currently mismatched: local says `itk`, HPC says `monai`).
- The `torch` CPU-build entry on HPC (required by MONAI transforms) must be explicitly noted in a comment in the local env file as intentionally absent locally.

### 5.2 SLURM Scripts

- All SLURM scripts must use the correct partition for their workload: CPU-only jobs on `cpu`, GPU jobs on `gpu-amd`.
- **Known bug:** `dataset_gen_128.slurm` requests partition `gpu-amd` for a CPU-only workload — must be changed to `cpu`.
- `start_script_rocm.slurm` must include an explicit `--mem` directive; defaulting to a per-CPU allocation is insufficient for large 3D volume training.
- All scripts must have a meaningful `--job-name`; `teste_sbatch` in `test_rocm.slurm` is a stale placeholder and must be replaced.
- Wall times must reflect actual job duration: the 6-hour wall time in `test_rocm.slurm` (a seconds-long smoke test) must be reduced.
- Stdout and stderr handling must be consistent; diagnostic jobs should merge both into a single timestamped log.
- Conda activation must be consistent across scripts; hardcoding `$HOME/miniconda3` is fragile — use the multi-path fallback pattern from `diagnose_rocm.slurm`.
- `dataset_gen_128.slurm` must make `VARIANTS` configurable via env var, matching `dataset_gen.slurm`.
- `dataset_gen_128.slurm` must add a version verification block (monai/nibabel/numpy) for reproducibility parity.

### 5.3 ROCm / Environment Diagnostics

- The three current Python diagnostic scripts (`check_kfd_version.py`, `diagnose_rocm.py`, `test_rocm.py`) must be merged into a single `check_environment.py` with a `--sections` CLI (`--kfd`, `--disk`, `--env`, `--torch`, `--all`).
- `--all` must reproduce the full combined output of all three current scripts with no loss of coverage.
- The KFD ioctl + HSA library `strings` scan from `check_kfd_version.py` must be retained as it is unique.
- The merged script must be runnable both locally (without a GPU) and via a single SLURM wrapper replacing the current two wrappers.

### 5.4 Data Transfer

- `transfer_nifti.sh` must replace `scp -r` with `rsync -aP` to make transfers resumable.
- Both transfer scripts must include a zero-file guard (exit with error if no local files match the pattern) — currently missing from `transfer_nifti.sh`.
- Both scripts must perform post-transfer file-count verification and exit non-zero on mismatch.
- Once both scripts use identical `rsync` logic, they may be unified under a single script with a `--mode nifti|zips` flag.
- SSH keepalive options (`ServerAliveInterval`, `ServerAliveCountMax`) and the cluster alias must be defined in one shared location (e.g. `~/.ssh/config`) rather than duplicated across scripts.

### 5.5 Cluster Workflow

- The standard deploy cycle (local edit → `ssh drummond` → `git pull` → `sbatch`) must be documented and optionally wrapped in a Makefile target.
- A `make fetch` target (or equivalent) must pull results from the cluster using `rsync`; see section 4.3 for sync filter requirements.
- SLURM `.out` log files must be written to (or copied into) the relevant experiment output directory, not left in the submission directory.
- A `make status` or equivalent must show running/pending jobs for the project without manual `squeue` invocations.
