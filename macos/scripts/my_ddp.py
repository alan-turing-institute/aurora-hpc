"""Fine tune Aurora weather model."""

print("importing...")
import os
from pathlib import Path

import torch
import torch.nn as nn
from aurora_loss import mae
from load_data import load_data
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from aurora import Aurora
from dataset import AuroraDataset

# PMI_RANK set by mpirun
RANK = os.environ["PMI_RANK"]
os.environ["RANK"] = RANK

# PMI_SIZE set by mpirun
WORLD_SIZE = os.environ["PMI_SIZE"]
assert WORLD_SIZE == "2"
os.environ["WORLD_SIZE"] = WORLD_SIZE

os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "29876"
USE_SUBDEVICES = os.environ.get("USE_SUBDEVICES", False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(5000, 5000)
        self.l2 = nn.Linear(5000, 5000)
        self.l3 = nn.Linear(5000, 5000)

    def forward(self, input):
        return self.l3(self.l2(self.l1(input)))


def main():
    print("Initialising process group with backend", "ccl", flush=True)

    world_size = os.environ["WORLD_SIZE"]
    rank = os.environ["RANK"]

    # ToDo Run 2 or more processes.
    init_process_group(
        world_size=int(world_size),
        rank=int(rank),
        backend="gloo",
    )

    device = f"xpu:{RANK}"
    print(f"Using {device=}")

    print("loading model...")
    model = Aurora(
        use_lora=False,  # Model was not fine-tuned.
        autocast=True,  # Use AMP.
    )
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    download_path = Path("../../era5/era_v_inf")

    print("loading data...")

    # 1 for RANK 0 and 3 for RANK 1.
    i = (int(RANK) * 2) + 1
    print(f"batching with {i=}")

    print("preparing model...")
    model.configure_activation_checkpointing()
    model = DDP(model).to(device)
    model.train()

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    dataset = AuroraDataset(
            data_path=download_path,
            t=1,
            static_filepath=Path("static.nc"),
            surface_filepath=Path("2023-01-01-surface-level.nc"),
            atmos_filepath=Path("2023-01-01-atmospheric.nc"),
        )
    sampler = DistributedSampler(dataset) if False else None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # We only have one batch.
        shuffle=False,  # We don't need to shuffle.
        sampler=sampler,
    )

    for epoch, (X, y) in enumerate(data_loader):
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

        print("performing backward pass...")
        loss.backward()

        print("optimizing...")
        optimizer.step()

    print("done")


main()
