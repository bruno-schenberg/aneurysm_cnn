# Infrastructure Review

## Diagnostics Scripts

### What each script does

**`check_kfd_version.py`**
Checks for a KFD kernel/userspace ABI version mismatch. It does three things: opens `/dev/kfd` via `ioctl` to read the kernel-reported KFD ABI version; searches the ROCm install tree for `kfd_ioctl.h` and prints the version constants defined there; and uses `ldd` on `rocminfo` to find `libhsa-runtime64`, then runs `strings` on it to extract any embedded version references. The output is a side-by-side comparison of what the kernel claims vs. what the installed userspace expects. ROCM_PATH is hardcoded to `/opt/ohpc/pub/apps/rocm/rocm-6.4.3`.

**`diagnose_rocm.py`**
A broad Python-level ROCm diagnostic that collects environment state in several sections: Python executable and version; `/dev/kfd` accessibility; KFD kernel driver version via sysfs; ROCm directory presence on disk; ROCm-related environment variables (`ROCM_PATH`, `LD_LIBRARY_PATH`, `PATH`); and PyTorch import with `torch.cuda.is_available()` and device count. Intended to be invoked from within a Conda environment so the PyTorch check is meaningful.

**`test_rocm.py`**
Minimal four-line GPU smoke test. Prints PyTorch version, `torch.cuda.is_available()`, `ROCR_VISIBLE_DEVICES`, and a list of visible GPU device names. No error handling — if `torch` is not importable, it raises immediately.

**`diagnose_rocm.slurm`**
Full SLURM job wrapper for a comprehensive diagnostic session intended to be sent to HPC support. It runs in sequence: module listing and `module show rocm/6.4.3`; ROCm disk layout checks (`/opt/rocm*`); device file and group checks (`/dev/kfd`, `/dev/dri/renderD*`); KFD sysfs version; `rocm-smi`; `rocminfo` under both `rocm/6.4.3` and `rocm/6.0.1` (two separate module loads); and finally activates the `aneurysm_cnn_rocm` Conda environment and runs `diagnose_rocm.py`. Stdout and stderr are merged into a single log file (`diagnose_rocm_<jobid>.log`). Resource request: 4 CPUs, 1 GPU, 10-minute wall time.

**`test_rocm.slurm`**
Lighter SLURM job wrapper focused on verifying that PyTorch can see GPUs at job runtime. It copies `test_rocm.py` to a per-job scratch directory, loads `rocm/6.0.1`, sets `HSA_OVERRIDE_GFX_VERSION=9.0.10`, activates the Conda environment, runs `rocm-smi` via `srun`, then runs `test_rocm.py` via `srun`, and cleans up scratch on exit. Stdout and stderr go to separate files (`%x_%j.out` / `%x_%j.err`). Resource request: 8 CPUs, 1 GPU, 6-hour wall time.

---

### Overlap analysis

There is significant overlap across all three Python scripts and between the two SLURM wrappers.

| Check | `check_kfd_version.py` | `diagnose_rocm.py` | `test_rocm.py` |
|---|---|---|---|
| `/dev/kfd` accessibility | yes (ioctl) | yes (open) | no |
| KFD kernel ABI version | yes (ioctl + sysfs) | yes (sysfs only) | no |
| ROCm disk layout | yes (kfd_ioctl.h search) | yes (ls /opt/rocm*) | no |
| Environment variables | no | yes | yes (ROCR_VISIBLE_DEVICES) |
| PyTorch available | no | yes | yes |
| GPU device names | no | no | yes |
| KFD userspace version | yes (ldd + strings) | no | no |

`diagnose_rocm.py` and `check_kfd_version.py` both check `/dev/kfd` and the KFD kernel version; they approach it differently (sysfs vs. ioctl) but report the same fact. `diagnose_rocm.py` and `test_rocm.py` both check `torch.cuda.is_available()` and device count; `test_rocm.py` adds device names but drops all other context.

The two SLURM wrappers overlap on: loading a ROCm module, running `rocm-smi`, and verifying PyTorch GPU access. `diagnose_rocm.slurm` is a superset of `test_rocm.slurm` in terms of what it checks — the only thing `test_rocm.slurm` adds that the other lacks is the scratch-directory copy pattern and the separate stdout/stderr files.

---

### Can they be merged into a single `check_environment.py`?

Yes, with a straightforward CLI. The three Python scripts cover complementary ground and have no conflicting logic. A single script with `--sections` flags (e.g. `--kfd`, `--env`, `--torch`, `--disk`) could replace all three without any loss of coverage. The KFD ioctl check from `check_kfd_version.py` is the only genuinely unique piece; everything else in `diagnose_rocm.py` could absorb `test_rocm.py` with one additional block printing device names.

A minimal CLI shape that would cover all current functionality:

```
python check_environment.py [--kfd] [--disk] [--env] [--torch] [--all]
```

`--all` would replicate the current combined output. Individual flags would let lightweight checks (e.g., just `--torch` for a quick smoke test) run without the heavier sysfs and `ldd`/`strings` probes.

`check_kfd_version.py` should be the one script kept as a standalone if merging is not done — it contains unique logic (ioctl + `strings` scan on the HSA library) not present in the other two.

---

### SLURM wrapper consistency

The two wrappers are inconsistent in several ways:

| Attribute | `diagnose_rocm.slurm` | `test_rocm.slurm` |
|---|---|---|
| `--job-name` | `diagnose_rocm` | `teste_sbatch` (stale placeholder name) |
| `--output` / `--error` | merged into one file | separate files |
| `--cpus-per-task` | 4 | 8 |
| `--time` | 0:10:00 | 6:00:00 |
| ROCm module loaded | `rocm/6.4.3` then `rocm/6.0.1` | `rocm/6.0.1` only |
| `HSA_OVERRIDE_GFX_VERSION` | set (under 6.4.3 section) | set |
| Conda activation method | tries three paths with `||` fallback | hardcodes `$HOME/miniconda3/bin/activate` |
| Scratch directory | not used | used (copy + cleanup) |
| Uses `srun` | no | yes (for `rocm-smi` and Python) |

Issues worth fixing if the wrappers are kept separate:
- `test_rocm.slurm` has `--job-name=teste_sbatch` — this is clearly a leftover from an earlier template.
- The 6-hour wall time in `test_rocm.slurm` is excessive for a script that finishes in seconds.
- Conda activation is inconsistent: `diagnose_rocm.slurm` tries multiple paths gracefully; `test_rocm.slurm` hardcodes `$HOME/miniconda3`.
- Merging stdout/stderr (as `diagnose_rocm.slurm` does) is the better default for diagnostic jobs — easier to read a single log in sequence.

If the Python scripts are merged, a single SLURM wrapper with a `--sections` argument passthrough would be the consistent counterpart, replacing both.

---

## Data & SLURM Scripts

### Script Inventory

#### transfer_nifti.sh
Transfers the local `/mnt/data/nifti` directory to the cluster (`drummond`) under `~/data/nifti` using `scp -r`. Before transferring it counts local `.nii.gz` files and compares against the remote count; if they match it exits early (idempotent). After transfer it re-counts and exits 1 on a mismatch. SSH keepalive options (`ServerAliveInterval=30`, `ServerAliveCountMax=20`) are set to prevent broken pipes on long transfers.

#### transfer_zips.sh
Transfers `.zip` archives from local `/mnt/data/raw` to cluster `~/data/raw_zips` using `rsync -aP`. Same cluster alias (`drummond`) and same SSH keepalive options. Uses rsync instead of scp, so partial transfers are resumable. Applies `--include="*.zip" --exclude="*"` to restrict transfer to zip files only. After transfer it verifies the file count. On success it prints next-step instructions for running `inventory_scan.py` on the cluster.

#### dataset_gen.slurm
Runs three sequential passes of `data_engine/dataset_gen.py` under SLURM partition `cpu`, one after the other in a single job:
- Pass 1: `128x128x128`, 4 workers
- Pass 2: `192x192x128`, 4 workers
- Pass 3: `256x256x176`, 2 workers (fewer workers due to larger volume size)

`INPUT_DIR`, `OUTPUT_BASE`, and `VARIANTS` are all configurable via environment variables with sane defaults. All three passes share the same conda environment (`aneurysm_cnn_data_hpc`). The job exits 1 if any pass fails. Wall time is set to 24 hours to cover all three sequential passes. Prints import versions (monai, nibabel, numpy) for reproducibility.

#### dataset_gen_128.slurm
Runs a single pass of `data_engine/dataset_gen.py` for `128x128x128` only, submitted to partition `gpu-amd` (not `cpu`). Hard-codes variants `C D A B E F` (not configurable via env var). No version verification block. Wall time 12 hours. Same conda env and same default `INPUT_DIR`/`OUTPUT_BASE`.

#### environment_data.yml (local)
Conda environment `aneurysm_cnn_data`. Specifies `python=3.11.15` (pinned patch version). Includes `pydicom=3.0.2`, `nilearn=0.13.1`, `itk=5.3.0`, `pytest=9.0.2`, `ruff=0.15.7` — none of which appear in the HPC env. No `torch` dependency (data pipeline does not need it locally). Uses `setuptools=69.5.1` pinned with a comment explaining itk compatibility.

#### environment_data_hpc.yml (HPC)
Conda environment `aneurysm_cnn_data_hpc`. Specifies `python=3.11` (unpinned minor). Installs `torch==2.1.0` via pip (CPU build). Missing all of `pydicom`, `nilearn`, `itk`, `pytest`, `ruff`. `setuptools` comment says the pin is for monai, whereas local says itk — different stated reasons for the same pin value. No `ruff`, no `pytest`.

#### start_script_rocm.slurm (training)
Runs training under partition `gpu-amd` with 8 CPUs and 1 GPU. Copies `training_engine/` to a per-job scratch directory (`/scratch/local/$SLURM_JOB_ID`) before running, then copies results back after. Loads `rocm/6.4.3` and sets `HSA_OVERRIDE_GFX_VERSION=9.0.10` for MI210 (gfx90a) compatibility. Activates `aneurysm_cnn_rocm`. The experiments file defaults to `experiments_256_baseline.json` and is overridable via `EXPERIMENTS_FILE`. Wall time is 80 hours. Cleans up scratch on exit.

---

### Transfer Scripts — Unification Assessment

The two scripts share the same structure (create remote dir -> count local -> skip if done -> transfer -> verify count -> exit on mismatch) and the same SSH options and cluster alias. The differences are:

| Property | transfer_nifti.sh | transfer_zips.sh |
|---|---|---|
| Transfer tool | `scp -r` | `rsync -aP` |
| File pattern | `*.nii.gz` (recursive) | `*.zip` (maxdepth 1) |
| Local dir | `/mnt/data/nifti` | `/mnt/data/raw` |
| Remote dir | `data/nifti` | `data/raw_zips` |
| On 0 local files | transfers 0 (no guard) | exits with error |
| Post-transfer hint | none | prints next-step instructions |

**Verdict: unification is feasible but brings marginal value.** The transfer tools differ (`scp` vs `rsync`), and the nifti script transfers recursively while the zips script is flat (`maxdepth 1`). A unified script would need a `--mode nifti|zips` flag or equivalent, adding branching that offsets the deduplication gain. The more impactful near-term fix is to replace `scp` in `transfer_nifti.sh` with `rsync` (resumable) and add a zero-file guard, bringing the two scripts to identical logic — at which point unification becomes straightforward.

---

### dataset_gen SLURM Scripts — Unification Assessment

`dataset_gen.slurm` already runs the 128 pass as its first of three passes. `dataset_gen_128.slurm` is a stripped-down single-pass variant added for faster iteration on the 128 shape alone.

Key divergences:

| Property | dataset_gen.slurm | dataset_gen_128.slurm |
|---|---|---|
| Partition | `cpu` | `gpu-amd` |
| Passes | 3 (128, 192, 256) | 1 (128 only) |
| VARIANTS | configurable via env var | hard-coded |
| Version verification | yes (monai/nibabel/numpy) | no |
| Wall time | 24 h | 12 h |
| Exit code propagation | explicit per-pass status | implicit |

**Verdict: do not unify mechanically, but align the two scripts.** They serve different purposes — full multi-shape generation vs. quick single-shape runs — and should coexist. However, `dataset_gen_128.slurm` uses partition `gpu-amd` for a CPU-only workload, which is incorrect. It should use `cpu`, matching `dataset_gen.slurm`. It should also make `VARIANTS` configurable and add the version verification block for consistency.

If a single-pass capability is needed on an ongoing basis, the cleaner long-term approach is to add a `TARGET_SHAPE` env var to `dataset_gen.slurm` so any shape can be submitted independently, eliminating `dataset_gen_128.slurm` entirely.

---

### Environment Sync — Drift Assessment

The local and HPC environments share the same pinned versions for the packages that appear in both (`numpy=1.26.4`, `nibabel=5.4.0`, `monai=1.5.2`, `setuptools=69.5.1`). Core package versions are in sync.

Drift and gaps:

1. **Python version pin**: local pins `python=3.11.15` (patch-level), HPC pins `python=3.11` (minor only). In practice conda will resolve these to the same interpreter on compatible platforms, but the inconsistency is a maintenance risk. Both should use the same level of pinning.

2. **Packages present locally but absent on HPC**: `pydicom`, `nilearn`, `itk`, `pytest`, `ruff`. If any data-engine script that imports `pydicom` or `nilearn` is expected to run on the HPC, the HPC env is incomplete. `pytest` and `ruff` are dev tools; their absence on HPC is expected and acceptable.

3. **`torch` only on HPC**: The local env does not install PyTorch. The HPC env installs a CPU build (`torch==2.1.0` via pip), presumably because MONAI's data transforms require it. This is intentional per the comment in the file but undocumented in the local env.

4. **`setuptools` comment discrepancy**: local says `# itk 5.3.x uses pkg_resources`; HPC says `# monai uses pkg_resources`. Both share the same pin for different stated reasons. The pin is correct in both cases, but the inconsistent justification is confusing and should be reconciled.

**Summary**: core package versions are in sync; the main drift is the set of packages (`pydicom`, `nilearn`, `itk`) present locally but not on HPC, and the python version pin granularity. No version conflicts detected.

---

### SLURM Resource Consistency

| Script | Partition | CPUs | GPUs | Memory | Wall time |
|---|---|---|---|---|---|
| dataset_gen.slurm | `cpu` | 4 | none | 48 G | 24 h |
| dataset_gen_128.slurm | `gpu-amd` | 4 | none | 48 G | 12 h |
| start_script_rocm.slurm | `gpu-amd` | 8 | 1 | unset | 80 h |

Issues:

- **`dataset_gen_128.slurm` uses `gpu-amd` but requests no GPUs.** Submitting a CPU-only job to the GPU partition wastes a scarce queue slot and may violate cluster policy. It should be changed to `cpu`.
- **`start_script_rocm.slurm` has no `--mem` directive.** On most SLURM clusters this defaults to a small per-CPU allocation, which may be insufficient for training large 3D volumes. An explicit memory request is recommended.
- CPU counts are consistent for data jobs (4 CPUs each). Training correctly uses 8 CPUs to feed PyTorch DataLoader workers.
- Both data scripts use 48 G memory; this is consistent and reasonable for in-memory NIfTI processing.
- Partition naming is consistent (`cpu` for data, `gpu-amd` for GPU training) except for the misclassified `dataset_gen_128.slurm`.
