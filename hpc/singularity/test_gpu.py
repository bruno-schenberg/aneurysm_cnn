import torch

print(f"PyTorch:       {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Device count:  {n}")
    for i in range(n):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected — check HSA_OVERRIDE_GFX_VERSION and --rocm flag.")
