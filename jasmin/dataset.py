#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute
import logging
import time
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import Dataset

from aurora import Batch, Metadata

BUILD_SURFACE_VAR_MAP = {
    "t2m": "tas",
    "u10": "uas",
    "v10": "vas",
    "msl": "psl",
}
BUILD_ATMOS_VAR_MAP = {
    "t": "ta",
    "u": "ua",
    "v": "va",
    "q": "hus",
    "z": "zg",
}
STANDARD_GRAVITY = 9.80665
LOGGER = logging.getLogger(__name__)


def open_build_variables(
    data_path: Path,
    variable_map: dict[str, str],
    use_dask: bool,
) -> xr.Dataset:
    """Open workflow build outputs and rename variables/coords for Aurora."""
    open_start = time.perf_counter()
    datasets = []
    for aurora_name, build_name in variable_map.items():
        variable_dir = data_path / build_name
        files = sorted(variable_dir.glob("*.regridded.nc"))
        if not files:
            raise FileNotFoundError(
                f"No regridded NetCDF files found for {build_name} in {variable_dir}"
            )
        variable_start = time.perf_counter()
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks={} if use_dask else None,
            engine="netcdf4",
        )
        LOGGER.info(
            "Opened %s NetCDF files for %s -> %s from %s in %.3fs",
            len(files),
            build_name,
            aurora_name,
            variable_dir,
            time.perf_counter() - variable_start,
        )
        ds = ds.rename({build_name: aurora_name})
        if aurora_name == "z" and build_name == "zg":
            ds[aurora_name] = ds[aurora_name] * STANDARD_GRAVITY
            ds[aurora_name].attrs.update(
                {
                    "units": "m2 s-2",
                    "standard_name": "geopotential",
                    "source_units": "m",
                    "source_standard_name": "geopotential_height",
                }
            )
        datasets.append(ds[[aurora_name]])

    combined = xr.merge(datasets, compat="override")
    combined = combined.rename(
        {
            name: renamed
            for name, renamed in {
                "lat": "latitude",
                "lon": "longitude",
                "time": "valid_time",
                "plev": "pressure_level",
            }.items()
            if name in combined.coords or name in combined.dims
        }
    )
    LOGGER.info(
        "Opened and merged build variables from %s in %.3fs: variables=%s",
        data_path,
        time.perf_counter() - open_start,
        list(combined.data_vars),
    )
    return combined


class AuroraDataset(Dataset):
    """Aurora dataset.

    Provides an indexable dataset of weather variables read in from disk.

    Args:
        data_path (Path): Directory to read in the data from.
        t (int): the number of additional timesteps to load alongside each datapoint.
        static_data (Path): Static NetCDF filename relative to `data_path`.
        use_dask (bool): Whether to use dask to load the datasets.
    """

    def __init__(
        self,
        data_path: str | Path,
        t: int,
        static_data: str | Path = Path("0pt25_static.nc"),
        use_dask: bool = False,
    ):
        self.t = t

        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(static_data, str):
            static_data = Path(static_data)

        surface_path = data_path / "build/surface.regridded"
        atmos_path = data_path / "build/atmos.regridded"
        static_path = data_path / static_data

        if not surface_path.is_dir():
            raise FileNotFoundError(f"Missing surface build directory: {surface_path}")
        if not atmos_path.is_dir():
            raise FileNotFoundError(
                f"Missing atmospheric build directory: {atmos_path}"
            )
        if not static_path.is_file():
            raise FileNotFoundError(f"Missing static NetCDF file: {static_path}")

        open_start = time.perf_counter()
        self.surf_vars_ds = self._open_build_variables(
            surface_path,
            BUILD_SURFACE_VAR_MAP,
            use_dask,
        )
        self.atmos_vars_ds = self._open_build_variables(
            atmos_path,
            BUILD_ATMOS_VAR_MAP,
            use_dask,
        )
        static_start = time.perf_counter()
        self.static_vars_ds = xr.open_dataset(
            static_path,
            engine="netcdf4",
            chunks={} if use_dask else None,
        )
        LOGGER.info(
            "Opened static NetCDF %s in %.3fs",
            static_path,
            time.perf_counter() - static_start,
        )

        tensor_start = time.perf_counter()
        self.surf_vars = {
            "2t": torch.from_numpy(self.surf_vars_ds["t2m"].values),
            "10u": torch.from_numpy(self.surf_vars_ds["u10"].values),
            "10v": torch.from_numpy(self.surf_vars_ds["v10"].values),
            "msl": torch.from_numpy(self.surf_vars_ds["msl"].values),
        }
        self.atmos_vars = {
            "t": torch.from_numpy(self.atmos_vars_ds["t"].values),
            "u": torch.from_numpy(self.atmos_vars_ds["u"].values),
            "v": torch.from_numpy(self.atmos_vars_ds["v"].values),
            "q": torch.from_numpy(self.atmos_vars_ds["q"].values),
            "z": torch.from_numpy(self.atmos_vars_ds["z"].values),
        }
        self.static_vars = {
            "z": self._static_tensor("z"),
            "slt": self._static_tensor("slt"),
            "lsm": self._static_tensor("lsm"),
        }
        self.lat = torch.from_numpy(self.surf_vars_ds.latitude.values)
        self.lon = torch.from_numpy(self.surf_vars_ds.longitude.values)
        self.valid_time = self.surf_vars_ds.valid_time.values.astype(
            "datetime64[s]"
        ).tolist()
        self.atmos_levels = self._atmos_levels()
        self.length = len(self.surf_vars["2t"]) - self.t - 1

        for ds in (self.surf_vars_ds, self.atmos_vars_ds, self.static_vars_ds):
            ds.close()
        del self.surf_vars_ds
        del self.atmos_vars_ds
        del self.static_vars_ds

        LOGGER.info(
            "Materialized AuroraDataset tensors in %.3fs: surface=%s atmos=%s static=%s",
            time.perf_counter() - tensor_start,
            list(self.surf_vars),
            list(self.atmos_vars),
            list(self.static_vars),
        )
        LOGGER.info(
            "Initialized AuroraDataset from %s in %.3fs: timesteps=%s samples=%s history=%s",
            data_path,
            time.perf_counter() - open_start,
            len(self.valid_time),
            self.length,
            self.t + 1,
        )

    def _open_build_variables(
        self,
        data_path: Path,
        variable_map: dict[str, str],
        use_dask: bool,
    ) -> xr.Dataset:
        return open_build_variables(data_path, variable_map, use_dask)

    def _get_batch(self, timerange):
        """Returns a batch covering a time range.

        Args:
            timerange (list): the range of values over time to return in the batch.
        """

        batch = Batch(
            surf_vars={
                # First select time points `index` and `index - 1`. Afterwards, `[None]` inserts a
                # batch dimension of size one.
                "2t": self.surf_vars["2t"][timerange][None],
                "10u": self.surf_vars["10u"][timerange][None],
                "10v": self.surf_vars["10v"][timerange][None],
                "msl": self.surf_vars["msl"][timerange][None],
            },
            static_vars={
                # The static variables are constant, so we just get them for the first time.
                "z": self.static_vars["z"],
                "slt": self.static_vars["slt"],
                "lsm": self.static_vars["lsm"],
            },
            atmos_vars={
                "t": self.atmos_vars["t"][timerange][None],
                "u": self.atmos_vars["u"][timerange][None],
                "v": self.atmos_vars["v"][timerange][None],
                "q": self.atmos_vars["q"][timerange][None],
                "z": self.atmos_vars["z"][timerange][None],
            },
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                # `datetime.datetime`s.
                # https://microsoft.github.io/aurora/batch.html#batch-metadata
                # Note that this needs to be a tuple of length one:
                # one value for every batch element.
                time=(self.valid_time[timerange[-1]],),
                atmos_levels=self.atmos_levels,
            ),
        )

        return batch

    def _static_tensor(self, name: str) -> torch.Tensor:
        values = self.static_vars_ds[name].values
        if values.ndim == 3:
            values = values[0]
        return torch.from_numpy(values)

    def _atmos_levels(self) -> tuple[int, ...]:
        levels = self.atmos_vars_ds.pressure_level.values
        if levels.max() > 2000:
            levels = levels / 100
        return tuple(int(level) for level in levels)

    def __getitem__(self, index):
        """Returns input and target batches for the given index.

        Args:
            index (int): the index of the batch to retreive.
        """
        timerange = [t + index for t in range(self.t + 1)]
        LOGGER.debug("Loading dataset index=%s input_timerange=%s", index, timerange)
        inputs = self._get_batch(timerange)
        # In case the `t` dimentions is needed for comparison with the output of the model
        # target = self._get_batch(index, [self.t + 1])
        target = self._get_batch([timerange[-1] + 1])
        LOGGER.debug("Loaded dataset index=%s input_timerange=%s", index, timerange)

        LOGGER.warning(
            "lon dtype=%s min=%s max=%s",
            inputs.metadata.lon.dtype,
            inputs.metadata.lon.min().item(),
            inputs.metadata.lon.max().item(),
        )
        return inputs, target

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
