#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import Dataset

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
        use_dask (bool): Whether to use dask to load the datasets.
    """

    def __init__(
        self,
        data_path: str | Path,
        t: int,
        static_filepath: str | Path = Path("static.nc"),
        surface_filepath: str | Path = Path("2023-01-01-surface-level.nc"),
        atmos_filepath: str | Path = Path("2023-01-01-atmospheric.nc"),
        use_dask: bool = False,
    ):
        self.t = t

        # Convert string paths to Path objects if necessary
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(static_filepath, str):
            static_filepath = Path(static_filepath)
        if isinstance(surface_filepath, str):
            surface_filepath = Path(surface_filepath)
        if isinstance(atmos_filepath, str):
            atmos_filepath = Path(atmos_filepath)

        self.static_vars_ds = xr.open_dataset(
            data_path / static_filepath,
            engine="netcdf4",
            chunks={} if use_dask else None,
        )
        self.surf_vars_ds = xr.open_dataset(
            data_path / surface_filepath,
            engine="netcdf4",
            chunks={} if use_dask else None,
        )
        self.atmos_vars_ds = xr.open_dataset(
            data_path / atmos_filepath,
            engine="netcdf4",
            chunks={} if use_dask else None,
        )
        self.length = (
            len(torch.from_numpy(self.surf_vars_ds["t2m"].values)) - self.t - 1
        )

    def _get_batch(self, timerange):
        """Returns a batch covering a time range.

        Args:
            timerange (list): the range of values over time to return in the batch.
        """
        return Batch(
            surf_vars={
                # First select time points `index` and `index - 1`. Afterwards, `[None]` inserts a
                # batch dimension of size one.
                "2t": torch.from_numpy(
                    self.surf_vars_ds["t2m"].values[timerange][None]
                ),
                "10u": torch.from_numpy(
                    self.surf_vars_ds["u10"].values[timerange][None]
                ),
                "10v": torch.from_numpy(
                    self.surf_vars_ds["v10"].values[timerange][None]
                ),
                "msl": torch.from_numpy(
                    self.surf_vars_ds["msl"].values[timerange][None]
                ),
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
                # `datetime.datetime`s.
                # https://microsoft.github.io/aurora/batch.html#batch-metadata
                # Note that this needs to be a tuple of length one:
                # one value for every batch element.
                time=(
                    self.surf_vars_ds.valid_time.values.astype(
                        "datetime64[s]"
                    ).tolist()[timerange[-1]],
                ),
                atmos_levels=tuple(
                    int(level) for level in self.atmos_vars_ds.pressure_level.values
                ),
            ),
        )

    def __getitem__(self, index):
        """Returns input and target batches for the given index.

        Args:
            index (int): the index of the batch to retreive.
        """
        timerange = [t + index for t in range(self.t + 1)]
        input = self._get_batch(timerange)
        # In case the `t` dimentions is needed for comparison with the output of the model
        # target = self._get_batch(index, [self.t + 1])
        target = self._get_batch([timerange[-1] + 1])
        return input, target

    def __len__(self):
        """Returns the total number of batches available."""
        return self.length


def batch_collate_fn(batches):
    """Collate a list of batches into a single batch.

    Args:
        batches ([Batch, Batch,...]): A list of batches to collate into a single
            batch.

    Returns:
        batch (Batch): A single batch containing all of the data.
    """

    # Start with the first batch
    result = Batch(
        batches[0].surf_vars,
        batches[0].static_vars,
        batches[0].atmos_vars,
        batches[0].metadata,
    )
    # Append the other batches to it

    # Surface variables
    keys = result.surf_vars.keys()
    # Merge the tensors along the batch dimension
    for key in keys:
        for idx in range(1, len(batches)):
            result.surf_vars[key] = torch.cat(
                [result.surf_vars[key], batches[idx].surf_vars[key]], 0
            )

    # Static variables remain constant
    result.static_vars = batches[0].static_vars

    # Atmospheric variables
    keys = result.atmos_vars.keys()
    # Merge the tensors along the batch dimension
    for key in keys:
        for idx in range(1, len(batches)):
            result.atmos_vars[key] = torch.cat(
                [result.atmos_vars[key], batches[idx].atmos_vars[key]], 0
            )

    # Metadata
    result.metadata.time = [t for item in batches for t in item.metadata.time]

    return result


def aurora_collate_fn(data):
    """Collate a list of (input, output) batch pairs into a single batch pair.

    Provides a collate_fn for batch collation during training. See:
    https://docs.pytorch.org/docs/stable/data.html#working-with-collate-fn

    Apparently this only works with a batch size of 1, which undermines its
    value to a large extent. This may be a limitation of the Aurora model, or
    it could be that this has been implemented incrrecoty; I'm not certain at
    present.

    Setting a batch size of None will prevent this function from being used.

    Args:
        batch ([(Batch, Batch),...]): A list of (input, output) batch pairs to
            collate into a single batch pair
    Returns:
        batch ((Batch, Batch)): A single (input, output) batch pair
            containing all of the data.
    """

    # Input type is [(Batch, Batch),...] where the list contains batch_size elements
    # Return type is (Batch, Batch)
    X, y = zip(*data)
    return (batch_collate_fn(X), batch_collate_fn(y))
