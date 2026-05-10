"""
check_kfd_version.py

Prints the KFD kernel ABI version and the version expected by the
installed ROCm user-space, so version mismatches are explicit.
"""

import fcntl
import os
import struct
import subprocess

ROCM_PATH = "/opt/ohpc/pub/apps/rocm/rocm-6.4.3"
AMDKFD_IOC_GET_VERSION = 0xC0084B01

print("=== KFD version check ===")

# 1. Kernel ABI version (what the driver reports)
try:
    fd = os.open("/dev/kfd", os.O_RDWR)
    buf = bytearray(8)
    fcntl.ioctl(fd, AMDKFD_IOC_GET_VERSION, buf)
    major, minor = struct.unpack("II", buf)
    print(f"Kernel KFD ABI version : {major}.{minor}")
    os.close(fd)
except Exception as e:
    print(f"Failed to query /dev/kfd: {e}")

# 2. Expected version from any kfd_ioctl.h header found under ROCm path
print(f"\nSearching for kfd_ioctl.h under {ROCM_PATH} ...")
result = subprocess.run(
    ["find", ROCM_PATH, "-name", "kfd_ioctl.h"],
    capture_output=True, text=True
)
headers = result.stdout.strip().splitlines()
if headers:
    for h in headers:
        print(f"Found header: {h}")
        try:
            with open(h) as f:
                for line in f:
                    if "KFD_IOCTL_MAJOR_VERSION" in line or "KFD_IOCTL_MINOR_VERSION" in line:
                        print(f"  {line.strip()}")
        except PermissionError:
            print(f"  (permission denied)")
else:
    print("No kfd_ioctl.h found under ROCm path.")

# 3. Find the actual HSA library path via ldd, then scan it
print("\nLocating libhsa-runtime64 via ldd ...")
rocminfo_path = subprocess.run(["which", "rocminfo"], capture_output=True, text=True).stdout.strip()
hsa_lib = None
if rocminfo_path:
    ldd = subprocess.run(["ldd", rocminfo_path], capture_output=True, text=True)
    for line in ldd.stdout.splitlines():
        if "libhsa-runtime64" in line and "=>" in line:
            hsa_lib = line.split("=>")[1].split("(")[0].strip()
            print(f"Found: {hsa_lib}")
            break

if hsa_lib and os.path.exists(hsa_lib):
    print("Scanning for version strings ...")
    result = subprocess.run(["strings", hsa_lib], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        l = line.strip()
        if "KFD_IOCTL" in l or "kfd_ioctl" in l or "major_version" in l or "minor_version" in l:
            print(f"  {l}")
    # Also look for version number patterns like "1.17" or "1.18"
    import re
    for line in result.stdout.splitlines():
        if re.fullmatch(r"1\.\d{1,2}", line.strip()):
            print(f"  version string: {line.strip()}")
else:
    print("Could not locate libhsa-runtime64 — is rocm/6.4.3 module loaded?")
