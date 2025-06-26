#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
from pathlib import Path
from torch.utils.data import Dataset
import xarray as xr
import torch
from aurora import Batch, Metadata

class AuroraDataset(Dataset):
    """Aurora dataset.

    Provides an indexable dataset of ERA5 weather variables read in from file.

    Args:
        data_path (Path): Directory to read in the data from.
        t (int): the number of additional timesteps to load alongside each datapoint.
        static_filepath (Path): file containing the static variable data, relative to `data_path`.
        surface_filepath (Path): file containing the surface-level variable data, relative to `data_path`.
        atmos_filepath (Path): file containing the atmospheric variable data, relative to `data_path`.
    """
    def __init__(
            self,
            data_path: Path,
            t: int,
            static_filepath = Path("static.nc"),
            surface_filepath = Path("2023-01-01-surface-level.nc"),
            atmos_filepath = Path("2023-01-01-atmospheric.nc"),
            ):
        self.t = t
        self.static_vars_ds = xr.open_dataset(data_path / static_filepath, engine="netcdf4")
        self.surf_vars_ds = xr.open_dataset(data_path / surface_filepath, engine="netcdf4")
        self.atmos_vars_ds = xr.open_dataset(data_path / atmos_filepath, engine="netcdf4")
        self.length = len(torch.from_numpy(self.surf_vars_ds["t2m"].values)) - self.t

    def __getitem__(self, index):
        timerange = [t + index for t in range(self.t + 1)]
        input = Batch(
            surf_vars={
                # First select time points `index` and `index - 1`. Afterwards, `[None]` inserts a
                # batch dimension of size one.
                "2t": torch.from_numpy(self.surf_vars_ds["t2m"].values[timerange][None]),
                "10u": torch.from_numpy(self.surf_vars_ds["u10"].values[timerange][None]),
                "10v": torch.from_numpy(self.surf_vars_ds["v10"].values[timerange][None]),
                "msl": torch.from_numpy(self.surf_vars_ds["msl"].values[timerange][None]),
            },
            static_vars={
                # The static variables are constant, so we just get them for the first time.
                "z": torch.from_numpy(self.static_vars_ds["z"].values[0]),
                "slt": torch.from_numpy(self.static_vars_ds["slt"].values[0]),
                "lsm": torch.from_numpy(self.static_vars_ds["lsm"].values[0]),
            },
            atmos_vars={
                "t": torch.from_numpy(self.atmos_vars_ds["t"].values[timerange][None]),
                "u": torch.from_numpy(self.atmos_vars_ds["u"].values[timerange][None]),
                "v": torch.from_numpy(self.atmos_vars_ds["v"].values[timerange][None]),
                "q": torch.from_numpy(self.atmos_vars_ds["q"].values[timerange][None]),
                "z": torch.from_numpy(self.atmos_vars_ds["z"].values[timerange][None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(self.surf_vars_ds.latitude.values),
                lon=torch.from_numpy(self.surf_vars_ds.longitude.values),
                # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                # `datetime.datetime`s. Note that this needs to be a tuple of length one:
                # one value for every batch element.
                time=(self.surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[index],),
                atmos_levels=tuple(int(level) for level in self.atmos_vars_ds.pressure_level.values),
            ),
        )
        target = Batch(
            surf_vars={
                # First select time points `index` and `index - 1`. Afterwards, `[None]` inserts a
                # batch dimension of size one.
                "2t": torch.from_numpy(self.surf_vars_ds["t2m"].values[timerange][None]),
                "10u": torch.from_numpy(self.surf_vars_ds["u10"].values[timerange][None]),
                "10v": torch.from_numpy(self.surf_vars_ds["v10"].values[timerange][None]),
                "msl": torch.from_numpy(self.surf_vars_ds["msl"].values[timerange][None]),
            },
            static_vars={
                # The static variables are constant, so we just get them for the first time.
                "z": torch.from_numpy(self.static_vars_ds["z"].values[0]),
                "slt": torch.from_numpy(self.static_vars_ds["slt"].values[0]),
                "lsm": torch.from_numpy(self.static_vars_ds["lsm"].values[0]),
            },
            atmos_vars={
                "t": torch.from_numpy(self.atmos_vars_ds["t"].values[timerange][None]),
                "u": torch.from_numpy(self.atmos_vars_ds["u"].values[timerange][None]),
                "v": torch.from_numpy(self.atmos_vars_ds["v"].values[timerange][None]),
                "q": torch.from_numpy(self.atmos_vars_ds["q"].values[timerange][None]),
                "z": torch.from_numpy(self.atmos_vars_ds["z"].values[timerange][None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(self.surf_vars_ds.latitude.values),
                lon=torch.from_numpy(self.surf_vars_ds.longitude.values),
                # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                # `datetime.datetime`s. Note that this needs to be a tuple of length one:
                # one value for every batch element.
                time=(self.surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[index],),
                atmos_levels=tuple(int(level) for level in self.atmos_vars_ds.pressure_level.values),
            ),
        )
        return batch

    def __len__(self):
        return self.length
