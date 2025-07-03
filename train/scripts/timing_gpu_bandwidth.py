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
    match the_device:
        case Device.CPU:
            torch_device = "cpu"
            synchronize = lambda: None
        case Device.XPU:
            import intel_extension_for_pytorch as ipex

            torch_device = "xpu"
            synchronize = torch.xpu.synchronize
        case Device.GPU:
            torch_device = "cuda"
            synchronize = torch.cuda.synchronize
        case _:
            raise RuntimeError()

    for size_mib in tqdm(SIZES_IN_MIB):
        num_elements = size_mib * 1024 // 4  # float32 = 4 bytes

        cpu_tensor = torch.randn((1024, num_elements), dtype=torch.float32)

        intended_size = size_mib * 1024 * 1024
        actual_size = cpu_tensor.element_size() * cpu_tensor.nelement()
        assert (
            intended_size == actual_size
        ), f"Wanted {intended_size}, got {actual_size}"

        # Warm up the GPU.
        _ = torch.randn(1, device=torch_device)

        # A closure to time.
        def transfer():
            synchronize()
            moved_tensor = cpu_tensor.to(torch_device, non_blocking=False)
            synchronize()

        timer = timeit.Timer(stmt=transfer)

        times = np.array(timer.repeat(repeat=REPEATS, number=1))

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
    parser.add_argument(
        "--device", type=Device, choices=[x.value for x in list(Device)], required=True
    )
    args = parser.parse_args()
    main(the_device=args.device)
