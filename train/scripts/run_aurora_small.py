"""Fine tune Aurora weather model."""

print("importing...")
import os
import pickle
import time
from pathlib import Path

import intel_extension_for_pytorch as ipex
import torch
import torch.nn as nn

# from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, DistributedSampler

from aurora import AuroraSmall
from aurora_hpc.aurora_loss import mae
from aurora_hpc.dataset import AuroraDataset, aurora_collate_fn

# os.environ["MASTER_ADDR"] = "0.0.0.0"
# os.environ["MASTER_PORT"] = "29876"
torch.use_deterministic_algorithms(True)


def main():
    print("seeding")
    torch.xpu.random.manual_seed_all(0)
    time_start_total = time.time()

    # init_process_group(
    #    world_size=1,
    #    rank=0,
    #    backend="gloo",
    # )

    device = "xpu"
    print(f"Using {device=}")

    print("loading model...")
    model = AuroraSmall()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

    download_path = Path("../../dawn/era5/era_v_inf")

    print("preparing model...")
    model.configure_activation_checkpointing()
    # model.train()

    # AdamW, as used in the paper.
    # optimizer = torch.optim.AdamW(model.parameters())

    print("loading data...")
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_data=Path("static.nc"),
        surface_data=Path("2023-01-surface-level.nc"),
        atmos_data=Path("2023-01-atmospheric.nc"),
    )

    # sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # We only have one batch.
        shuffle=False,  # We don't need to shuffle.
        # sampler=sampler,
        collate_fn=aurora_collate_fn,
    )

    times = []

    time_start = time.time()
    preds_losses = []
    # only to check our code is correct
    # with open("preds_losses.pkl", "wb") as f:
    #    pickle.dump(preds_losses, f)

    for epoch, (X, y) in enumerate(data_loader):  # Only run 3 epochs for testing.
        print(f"epoch {epoch}...")

        # Not really necessary, for one forward pass.
        # optimizer.zero_grad()

        print("performing forward pass...")
        pred = model(X)

        # space constraints
        # pred = pred.to("cpu")

        # mean absolute error of one variable
        print("calculating loss...")

        # Todo: Are pred's of type PyTree and does it matter?
        loss = mae(pred, y)

        time_end = time.time()
        times.append(time_end - time_start)
        time_start = time.time()

        with open(f"10v_{time_start}.pkl", "wb") as f:
            pickle.dump(pred.surf_vars["10v"], f)
        break

    # print("preds match:", preds_losses[0][0]==preds_losses[1][0])
    # print("losses match:", preds_losses[0][1]==preds_losses[1][1])

    # avg_time = sum(times[1:]) / len(times[1:])
    # print(f"Average time per epoch (ignoring first): {avg_time}")
    print(f"Total time for {len(times)} epochs: {sum(times)}")

    time_end_total = time.time()
    print(f"Total time: {time_end_total - time_start_total}")

    # destroy_process_group()
    print("done")


main()
