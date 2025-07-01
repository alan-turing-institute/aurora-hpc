"""Fine tune Aurora weather model."""

print("importing...")
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from aurora_loss import mae
from dataset import AuroraDataset, aurora_collate_fn
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, DistributedSampler

from aurora import AuroraSmall

os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "29876"


def main():
    start_time_total = time.time()

    init_process_group(
        world_size=1,
        rank=0,
        backend="gloo",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=}")

    print("loading model...")
    model = AuroraSmall()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

    download_path = Path("../../dawn/era5/era_v_inf")

    print("loading data...")

    print("preparing model...")
    model.configure_activation_checkpointing()
    model.train()

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_filepath=Path("static.nc"),
        surface_filepath=Path("2023-01-surface-level.nc"),
        atmos_filepath=Path("2023-01-atmospheric.nc"),
    )

    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # We only have one batch.
        shuffle=False,  # We don't need to shuffle.
        sampler=sampler,
        collate_fn=aurora_collate_fn,
    )

    times = []

    time_start = time.time()
    for epoch, (X, y) in enumerate(data_loader):  # Only run 3 epochs for testing.
        print(f"epoch {epoch}...")

        # Not really necessary, for one forward pass.
        optimizer.zero_grad()

        print("performing forward pass...")
        pred = model(X)

        # space constraints
        # pred = pred.to("cpu")

        # mean absolute error of one variable
        print("calculating loss...")

        # Todo: Are pred's of type PyTree and does it matter?
        loss = mae(pred, y)

        # print("performing backward pass...")
        # loss.backward()

        # print("optimizing...")
        # optimizer.step()

        time_end = time.time()
        times.append(time_end - time_start)
        time_start = time.time()

        if epoch == 2:
            print("Stopping after 3 epochs for testing.")
            break

    avg_time = sum(times[1:]) / len(times[1:])
    print(f"Average time per epoch (ignoring first): {avg_time}")
    print(f"Total time for {len(times)} epochs: {sum(times)}")

    end_time_total = time.time()
    print(f"Total time: {end_time_total - start_time_total}")

    destroy_process_group()
    print("done")


main()
