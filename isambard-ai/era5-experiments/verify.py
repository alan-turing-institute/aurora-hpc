import sys
import platform
import torch

print("--- PyTorch Verification ---")

# 1. Check Python and Platform Details
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor Architecture: {platform.machine()}")

# 2. Check PyTorch Installation
print(f"\nPyTorch Version: {torch.__version__}")
print(f"PyTorch is built with CUDA: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n\n!!! CRITICAL ERROR: PyTorch was NOT installed with CUDA support. !!!")
    print("Please check your installation command and NVIDIA driver.")
else:
    print("\n--- CUDA & GPU Details ---")
    # 3. Check CUDA Version PyTorch was built with
    print(f"PyTorch CUDA Version: {torch.version.cuda}")

    # 4. Check NVIDIA Driver Version (if accessible)
    # This might require the `pynvml` package: pip install pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        print(f"NVIDIA Driver Version: {driver_version}")
        pynvml.nvmlShutdown()
    except (ImportError, pynvml.NVMLError):
        print("NVIDIA Driver Version: Not accessible (pynvml not installed or failed to init).")

    # 5. Check GPU Details
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    if device_count > 0:
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            # For GH200, this should identify the Hopper H100 component.

    # 6. Simple CUDA Tensor Test
    print("\n--- Sanity Check: Tensor on GPU ---")
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"Successfully created a tensor on the GPU: {x}")
        print(f"Tensor device: {x.device}")
    except Exception as e:
        print(f"Failed to create a tensor on the GPU. Error: {e}")

print("\n--- Verification Complete ---")