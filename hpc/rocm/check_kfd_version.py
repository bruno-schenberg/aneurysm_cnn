"""
check_kfd_version.py

Prints the KFD kernel ABI version and the version expected by the
installed ROCm user-space, so version mismatches are explicit.
"""

import fcntl
import os
import struct

ROCM_PATH = "/opt/ohpc/pub/apps/rocm-6.4.3"
KFD_HEADER = f"{ROCM_PATH}/include/linux/kfd_ioctl.h"
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

# 2. Expected version (what ROCm user-space was compiled against)
try:
    with open(KFD_HEADER) as f:
        for line in f:
            if "KFD_IOCTL_MAJOR_VERSION" in line or "KFD_IOCTL_MINOR_VERSION" in line:
                print(f"ROCm expects           : {line.strip()}")
except FileNotFoundError:
    print(f"Header not found: {KFD_HEADER}")
