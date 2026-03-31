"""
check_kfd_version.py

Prints the KFD kernel ABI version and the version expected by the
installed ROCm user-space, so version mismatches are explicit.
"""

import fcntl
import os
import struct
import subprocess

ROCM_PATH = "/opt/ohpc/pub/apps/rocm-6.4.3"
HSA_LIB = f"{ROCM_PATH}/lib/libhsa-runtime64.so.1"
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

# 3. Search for version strings in the HSA runtime binary
print(f"\nSearching HSA runtime binary for version strings ...")
if os.path.exists(HSA_LIB):
    result = subprocess.run(
        ["strings", HSA_LIB],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "KFD_IOCTL" in line or "kfd_ioctl" in line or "major_version" in line:
            print(f"  {line.strip()}")
else:
    print(f"Library not found: {HSA_LIB}")
