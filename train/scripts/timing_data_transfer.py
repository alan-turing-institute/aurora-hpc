"""Move data to GPU."""

print("importing...")
import argparse
import time
from pathlib import Path

import torch
from dataset import AuroraDataset, aurora_collate_fn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--xpu", action="store_true", help="boolean of whether to use xpu")
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
    type=Path,
)
parser.add_argument("--dask", action="store_true", help="use dask for data loading")
parser.add_argument(
    "--num_workers", "-w", help="the number of data loader workers", default=0, type=int
)
args = parser.parse_args()

if args.xpu:
    import intel_extension_for_pytorch as ipex


def main(download_path: Path, use_dask: bool, num_workers: int, xpu: bool):
    time_start_total = time.time()

    print("loading data..." + " with dask" if use_dask else "")
    time_start_init_dataset = time.time()
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_filepath=Path("static.nc"),
        surface_filepath=Path("2023-01-surface-level.nc"),
        atmos_filepath=Path("2023-01-atmospheric.nc"),
        use_dask=use_dask,
    )
    time_end_init_dataset = time.time()
    print(
        f"Time to init AuroraDataset (loading metadata): {time_end_init_dataset - time_start_init_dataset}"
    )

    time_start_load_dataset = time.time()
    dataset.static_vars_ds.load()
    dataset.surf_vars_ds.load()
    dataset.atmos_vars_ds.load()
    time_end_load_dataset = time.time()
    print(f"Time to load datasets: {time_end_load_dataset - time_start_load_dataset}")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=False,  # We don't need to shuffle.
        sampler=None,
        collate_fn=aurora_collate_fn,
        num_workers=num_workers,
    )

    device = "xpu" if xpu else "cuda" if torch.cuda.is_available() else "cpu"

    times_ram = []
    time_gpu = []

    time_start_ram = time.time()
    for batch, data in enumerate(data_loader):
        print(f"batch {batch}...")
        X, y = data
        time_end_ram = time.time()
        times_ram.append(time_end_ram - time_start_ram)

        print("moving batch (input and target) to device")
        time_start_gpu = time.time()
        X = X.to(device)
        y = y.to(device)
        time_end_gpu = time.time()
        time_gpu.append(time_end_gpu - time_start_gpu)

        time_start_ram = time.time()

    print(f"Time for first batch (RAM): {times_ram[0]}")
    avg_time_ram = sum(times_ram[1:]) / len(times_ram[1:])
    print(f"Average time per batch (RAM, ignoring first): {avg_time_ram}")
    print(f"Total time for {len(times_ram)} batches (RAM): {sum(times_ram)}")

    print(f"Time for first batch (GPU): {time_gpu[0]}")
    avg_time_gpu = sum(time_gpu[1:]) / len(time_gpu[1:])
    print(f"Average time per batch (GPU, ignoring first): {avg_time_gpu}")
    print(f"Total time for {len(time_gpu)} batches (GPU): {sum(time_gpu)}")

    time_end_total = time.time()
    print(f"Total time: {time_end_total - time_start_total}")

    print("done")


main(args.download_path, args.dask, args.num_workers, args.xpu)
