#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import torch
import xarray as xr
import matplotlib.pyplot as plt

from aurora import AuroraSmall, rollout, Batch, Metadata
from pathlib import Path
from torch import nn, optim

# Fine-Tuning
# See https://microsoft.github.io/aurora/finetuning.html

print("loading model...")
model = AuroraSmall(
    use_lora=False,  # Model was not fine-tuned.
    autocast=True,  # Use AMP.
)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

# Data will be downloaded here.
download_path = Path("./downloads")
download_path = download_path.expanduser()

print("loading data...")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

i = 1  # Select this time index in the downloaded data.

print("batching...")
batch = Batch(
    surf_vars={
        # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
        # batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element.
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

print("preparing model...")
model = model.to("cuda")
model.train()
model.configure_activation_checkpointing()

# AdamW, as used in the paper.
optimizer = torch.optim.AdamW(model.parameters())

# Not really necessary, for one forward pass.
optimizer.zero_grad()

print("performing forward pass...")
pred = model.forward(batch)
#loss_fn = nn.CrossEntropyLoss()

# space constraints
pred = pred.to("cpu")

# mean absolute error of one variable
print("calculating loss...")
loss = torch.mean(torch.abs(pred.surf_vars["2t"] - batch.surf_vars["2t"][:,:,:720,:]))

print("performing backward pass...")
loss.backward()

print("optimizing...")
optimizer.step()

print("done")
