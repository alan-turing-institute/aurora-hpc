"""Fine tune Aurora weather model."""

print("importing...")
import os
from pathlib import Path

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # has side-effects
import torch
import torch.nn as nn
import xarray as xr
from aurora_loss import mae
from load_batches import get_gt_batch, get_input_batch
from load_data import load_data
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from aurora import Aurora, Batch, Metadata

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


def main():
    print("Initialising process group with backend", "ccl", flush=True)
    init_process_group(
        backend="ccl",
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

    # Load data
    static_vars_ds, surf_vars_ds, atmos_vars_ds = load_data(download_path)

    # Get input
    batch = get_input_batch(i, static_vars_ds, surf_vars_ds, atmos_vars_ds).to(device)

    # Get output
    ground_truth = get_gt_batch(i, static_vars_ds, surf_vars_ds, atmos_vars_ds).to(
        device
    )

    print("preparing model...")
    model.configure_activation_checkpointing()
    model = DDP(model).to(device)
    model.train()

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    for _ in range(2):
        # Not really necessary, for one forward pass.
        optimizer.zero_grad()

        print("performing forward pass...")
        pred = model.forward(batch)

        # space constraints
        # pred = pred.to("cpu")

        # mean absolute error of one variable
        print("calculating loss...")
        loss = mae(pred, ground_truth)

        print("performing backward pass...")
        loss.backward()

        print("optimizing...")
        optimizer.step()

    print("done")


main()
