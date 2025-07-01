"""Move data to GPU."""

print("importing...")
import argparse
import time
from pathlib import Path

from dataset import AuroraDataset, aurora_collate_fn
from torch.utils.data import DataLoader, DistributedSampler
import torch

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
    start_time_total = time.time()

    download_path = Path(download_path)

    print("loading data...")
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_filepath=Path("static.nc"),
        surface_filepath=Path("2023-01-surface-level.nc"),
        atmos_filepath=Path("2023-01-atmospheric.nc"),
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
    print(f"Average time per epoch (ignoring first): {avg_time}")
    print(f"Total time for {len(times)} epochs: {sum(times)}")

    end_time_total = time.time()
    print(f"Total time: {end_time_total - start_time_total}")

    print("done")


main(args.download_path, xpu=args.xpu)
