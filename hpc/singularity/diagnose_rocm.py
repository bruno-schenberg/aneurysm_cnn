"""
diagnose_rocm.py

Python-side ROCm diagnostic. Prints findings to stdout so the SLURM
script can capture everything in a single log file.
"""

import os
import subprocess
import sys


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (result.stdout + result.stderr).strip()
    print(out if out else "(no output)")


section("Python environment")
print(f"Python:      {sys.version}")
print(f"Executable:  {sys.executable}")

section("/dev/kfd access")
try:
    open("/dev/kfd", "rb").close()
    print("/dev/kfd: ACCESSIBLE")
except PermissionError as e:
    print(f"/dev/kfd: PERMISSION DENIED — {e}")
except FileNotFoundError:
    print("/dev/kfd: NOT FOUND")

section("KFD kernel driver version")
kfd_ver_paths = [
    "/sys/module/amdkfd/version",
    "/sys/module/amdkfd/parameters/minor_version",
]
for p in kfd_ver_paths:
    if os.path.exists(p):
        print(f"{p}: {open(p).read().strip()}")
    else:
        print(f"{p}: not present")
run("find /sys -name 'version' -path '*amdkfd*' 2>/dev/null")

section("ROCm installations on disk")
run("ls -la /opt/rocm* -d 2>&1")
run("readlink -f /opt/rocm 2>&1")

section("rocm/6.4.3 module paths (verify directory exists)")
for p in ["/opt/rocm-6.4.3/bin", "/opt/rocm-6.4.3/lib", "/opt/rocm-6.4.3/include"]:
    exists = os.path.isdir(p)
    print(f"  {'EXISTS' if exists else 'MISSING'}  {p}")

section("Active ROCm paths from environment")
for var in ["ROCM_PATH", "LD_LIBRARY_PATH", "PATH"]:
    val = os.environ.get(var, "(not set)")
    if var == "PATH":
        rocm_entries = [x for x in val.split(":") if "rocm" in x.lower()]
        print(f"PATH (rocm entries): {rocm_entries if rocm_entries else '(none)'}")
    else:
        print(f"{var}: {val}")

section("PyTorch import")
try:
    import torch  # noqa: E402
    print(f"torch version:       {torch.__version__}")
    print(f"torch.version.hip:   {getattr(torch.version, 'hip', 'N/A')}")
    print(f"cuda.is_available(): {torch.cuda.is_available()}")
    print(f"device_count():      {torch.cuda.device_count()}")
except ImportError as e:
    print(f"ImportError: {e}")
    print("(torch not installed in this environment — install with ROCm wheels to test)")

section("Done")
