"""Move data to GPU."""

print("importing...")
import argparse
import time
from pathlib import Path

import torch
from dataset import AuroraDataset, aurora_collate_fn
from torch.utils.data import DataLoader, DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--xpu", action="store_true", help="boolean of whether to use xpu")
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
)
args = parser.parse_args()

if args.xpu:
    import intel_extension_for_pytorch as ipex


def main(download_path: str, xpu: bool = False):
    time_start_total = time.time()

    download_path = Path(download_path)

    print("loading data...")
    time_start_load_xarray = time.time()
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_filepath=Path("static.nc"),
        surface_filepath=Path("2023-01-surface-level.nc"),
        atmos_filepath=Path("2023-01-atmospheric.nc"),
    )
    time_end_load_xarray = time.time()
    print(
        f"Time to init AuroraDataset: {time_end_load_xarray - time_start_load_xarray}"
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=False,  # We don't need to shuffle.
        sampler=None,
        collate_fn=aurora_collate_fn,
    )

    device = "xpu" if xpu else "cuda" if torch.cuda.is_available() else "cpu"

    times = []

    time_start = time.time()
    for batch, (X, y) in enumerate(data_loader):
        print(f"batch {batch}...")

        print("moving batch (input and target) to device")
        X.to(device)
        y.to(device)

        time_end = time.time()
        times.append(time_end - time_start)
        time_start = time.time()

    avg_time = sum(times[1:]) / len(times[1:])
    print(f"Time for first epoch: {times[0]}")
    print(f"Average time per epoch (ignoring first): {avg_time}")
    print(f"Total time for {len(times)} epochs: {sum(times)}")

    time_end_total = time.time()
    print(f"Total time: {time_end_total - time_start_total}")

    print("done")


main(args.download_path, xpu=args.xpu)
