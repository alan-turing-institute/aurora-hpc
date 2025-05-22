#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import torch
import xarray as xr
import matplotlib.pyplot as plt

from aurora import Aurora, rollout, Batch, Metadata
from pathlib import Path
from torch import nn, optim

# Fine-Tuning
# See https://microsoft.github.io/aurora/finetuning.html

# Data will be downloaded here.
download_path = Path("./downloads")
download_path = download_path.expanduser()

static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        #"10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        #"msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
        #"q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
        #"z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element. Select element 1, corresponding to time
        # 06:00.
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

# Load the model
model = Aurora(use_lora=False, autocast=False)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

# Perform fine-tuning
model = model.cuda()
model.train()
model.configure_activation_checkpointing()

optimizer = optim.AdamW(model.parameters())
optimizer.zero_grad()

with torch.autocast(device_type="cuda"):
    pred = model.forward(batch)
    pred = pred.to("cpu")
    batch = batch.to("cpu")
    loss = torch.mean(torch.abs(pred.surf_vars["2t"] - batch.surf_vars["2t"][:,:,:720,:]))

torch.cuda.empty_cache()

print(torch.cuda.memory_summary(device=None, abbreviated=False))

loss.backward()
optimizer.step()

