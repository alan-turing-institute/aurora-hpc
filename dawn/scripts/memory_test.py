print("Importing")
import torch
import intel_extension_for_pytorch as ipex

print("Imports done")


def test_gpu_memory(dtype=torch.float32):
    device = torch.device("xpu")
    print(f"Using device: {device}")

    # Start with 10MB and scale up
    size_mb = 10
    step_mb = 100
    max_allocated = 0

    try:
        tensors = []
        while True:
            # Each float32 = 4 bytes, so calculate number of elements
            num_elements = (size_mb * 1024 * 1024) // torch.tensor(
                [], dtype=dtype
            ).element_size()
            t = torch.empty(num_elements, dtype=dtype, device=device)
            tensors.append(t)
            max_allocated += size_mb
            print(f"Allocated ~{max_allocated} MB", end="\r")
            size_mb += step_mb
    except RuntimeError as e:
        print(f"\nOOM at ~{max_allocated} MB allocated")
        if "out of memory" in str(e).lower():
            pass
        else:
            raise e

    return max_allocated


if __name__ == "__main__":
    total = test_gpu_memory()
    print(f"Usable GPU memory is approximately: {total} MB")
