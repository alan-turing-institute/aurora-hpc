"""Move data to GPU."""

import argparse
import timeit
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm

SIZES_IN_MIB = [1, 4, 16, 64, 128, 256, 512, 1024, 10240]
REPEATS = 5

RESULTS = []


class Device(str, Enum):
    CPU = "cpu"
    XPU = "xpu"
    GPU = "gpu"


def main(the_device: Device):
    torch_device = "cpu"
    synchronize = lambda: None

    if the_device == Device.XPU:
        import intel_extension_for_pytorch as ipex

        torch_device = "xpu"
        synchronize = torch.xpu.synchronize
    elif the_device == Device.GPU:
        torch_device = "cude"
        synchronize = torch.cuda.synchronize

    for size_mib in tqdm(SIZES_IN_MIB):
        num_elements = size_mib * 1024 // 4  # float32 = 4 bytes

        cpu_tensor = torch.randn((1024, num_elements), dtype=torch.float32)

        # Warm up the GPU.
        _ = torch.randn(1, device=torch_device)

        # A closure to time.
        def transfer():
            synchronize()
            moved_tensor = cpu_tensor.to(torch_device, non_blocking=False)
            synchronize()

        timer = timeit.Timer(stmt=transfer)

        times = timer.repeat(repeat=REPEATS, number=1)

        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()
        bandwidth = size_mib / mean_time
        RESULTS.append((mean_time, std_time, bandwidth, size_mib))

    print("\nSummary of Transfer Performance:")
    print("=" * 80)
    header = f"{'Size (MiB)':>12} | {'Mean Time (ms)':>15} | {'Std Dev (ms)':>12} | {'Bandwidth (MiB/s)':>18}"
    print(header)
    print("-" * len(header))
    for mean_time, std_time, bandwidth, size_mib in RESULTS:
        print(
            f"{size_mib:12} | "
            f"{mean_time*1000:15.3f} | "
            f"{std_time*1000:12.3f} | "
            f"{bandwidth:18.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=Device, choices=[x.value for x in list(Device)])
    args = parser.parse_args()
    main(the_device=args.device)
