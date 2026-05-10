import torch
import os
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm/HIP Available: {torch.cuda.is_available()}")
print(f"ROCR_VISIBLE_DEVICES (from Python): {os.environ.get('ROCR_VISIBLE_DEVICES')}")
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPUs.")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs found.")