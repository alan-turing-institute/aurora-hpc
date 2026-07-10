"""Lightweight inference and evaluation helpers for the Aurora fine-tuning demo.

Unlike ``dataset.AuroraDataset`` (which materialises the whole month of data
as in-memory tensors, ~36 GB), these helpers keep the xarray datasets lazy and
only read the timesteps needed for the requested batch, so they are safe to
use interactively on a shared machine.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from dataset import (
    BUILD_ATMOS_VAR_MAP,
    BUILD_SURFACE_VAR_MAP,
    open_build_variables,
)

from aurora import Batch, Metadata

LOGGER = logging.getLogger(__name__)

# Batch key -> variable name in the opened surface dataset.
SURF_BATCH_FROM_DS = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
ATMOS_VARS = ("t", "u", "v", "q", "z")


class LazyCMIP6Data:
    """Lazily indexed view of the regridded CMIP6 build outputs.

    Args:
        data_path: Directory containing ``build/`` and the static NetCDF file,
            i.e. the same directory passed to ``train.py --data_path``.
        static_data: Static NetCDF filename relative to ``data_path``.
    """

    def __init__(
        self, data_path: str | Path, static_data: str | Path = "0pt25_static.nc"
    ):
        data_path = Path(data_path)
        self.surf_ds = open_build_variables(
            data_path / "build/surface.regridded", BUILD_SURFACE_VAR_MAP, use_dask=True
        )
        self.atmos_ds = open_build_variables(
            data_path / "build/atmos.regridded", BUILD_ATMOS_VAR_MAP, use_dask=True
        )
        static_ds = xr.open_dataset(data_path / Path(static_data), engine="netcdf4")
        self.static_vars = {
            name: self._static_tensor(static_ds, name) for name in ("z", "slt", "lsm")
        }
        static_ds.close()

        self.lat = torch.from_numpy(self.surf_ds.latitude.values)
        self.lon = torch.from_numpy(self.surf_ds.longitude.values)
        self.valid_time = self.surf_ds.valid_time.values.astype(
            "datetime64[s]"
        ).tolist()

        levels = self.atmos_ds.pressure_level.values
        if levels.max() > 2000:  # Pa -> hPa
            levels = levels / 100
        self.atmos_levels = tuple(int(level) for level in levels)

    @staticmethod
    def _static_tensor(static_ds: xr.Dataset, name: str) -> torch.Tensor:
        values = static_ds[name].values
        if values.ndim == 3:
            values = values[0]
        return torch.from_numpy(values)

    def __len__(self) -> int:
        return len(self.valid_time)

    def batch_at(self, index: int, t: int = 1) -> Batch:
        """Build an input batch whose history covers timesteps ``index .. index + t``.

        This matches the convention of ``AuroraDataset.__getitem__``: the model
        input at dataset index ``i`` covers times ``i`` and ``i + t``, and the
        prediction target is time ``i + t + 1``.
        """
        timerange = list(range(index, index + t + 1))
        return self._make_batch(timerange)

    def truth_at(self, index: int) -> Batch:
        """Single-timestep batch, e.g. as verification truth or a persistence forecast."""
        return self._make_batch([index])

    def _make_batch(self, timerange: list[int]) -> Batch:
        surf = self.surf_ds.isel(valid_time=timerange)
        atmos = self.atmos_ds.isel(valid_time=timerange)
        return Batch(
            surf_vars={
                key: torch.from_numpy(np.ascontiguousarray(surf[ds_name].values))[None]
                for key, ds_name in SURF_BATCH_FROM_DS.items()
            },
            static_vars=self.static_vars,
            atmos_vars={
                key: torch.from_numpy(np.ascontiguousarray(atmos[key].values))[None]
                for key in ATMOS_VARS
            },
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(self.valid_time[timerange[-1]],),
                atmos_levels=self.atmos_levels,
            ),
        )


def load_finetuned(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> torch.nn.Module:
    """Load fine-tuned weights from a ``train.py`` epoch checkpoint into ``model``.

    The checkpoint also contains ~10 GB of optimizer state; ``mmap=True`` keeps
    that on disk so only the model weights are materialised in memory.
    """
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def latitude_weights(lat: torch.Tensor) -> torch.Tensor:
    """cos(lat) area weights, normalised to mean 1."""
    weights = torch.cos(torch.deg2rad(lat.double()))
    return weights / weights.mean()


def _weighted_stats(diff: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    """Latitude-weighted error stats for ``diff`` with dims ``(..., lat, lon)``."""
    w = weights.view(*([1] * (diff.ndim - 2)), -1, 1)
    diff = diff.double()
    return {
        "rmse": torch.sqrt(torch.mean(w * diff**2)).item(),
        "mae": torch.mean(w * diff.abs()).item(),
        "bias": torch.mean(w * diff).item(),
    }


def compute_metrics(pred: Batch, truth: Batch) -> list[dict]:
    """Latitude-weighted RMSE/MAE/bias of ``pred`` against ``truth``.

    Returns one record per surface variable and one per (atmospheric variable,
    pressure level). The truth fields are cropped to the prediction's spatial
    shape, mirroring ``aurora_loss.mae``.
    """
    weights = latitude_weights(pred.metadata.lat.cpu())
    records = []
    for key, pred_field in pred.surf_vars.items():
        pred_field = pred_field.cpu()
        truth_field = truth.surf_vars[key][
            ..., : pred_field.shape[-2], : pred_field.shape[-1]
        ].cpu()
        records.append(
            {
                "var": key,
                "level": None,
                **_weighted_stats(pred_field - truth_field, weights),
            }
        )
    for key, pred_field in pred.atmos_vars.items():
        pred_field = pred_field.cpu()
        truth_field = truth.atmos_vars[key][
            ..., : pred_field.shape[-3], : pred_field.shape[-2], : pred_field.shape[-1]
        ].cpu()
        diff = pred_field - truth_field
        for i, level in enumerate(pred.metadata.atmos_levels):
            records.append(
                {
                    "var": key,
                    "level": level,
                    **_weighted_stats(diff[..., i, :, :], weights),
                }
            )
    return records


def surf_field(batch: Batch, var: str) -> np.ndarray:
    """Extract a surface field as a 2D numpy array (latest history step)."""
    return batch.surf_vars[var][0, -1].float().cpu().numpy()


def atmos_field(batch: Batch, var: str, level: int) -> np.ndarray:
    """Extract an atmospheric field at a pressure level as a 2D numpy array."""
    level_index = batch.metadata.atmos_levels.index(level)
    return batch.atmos_vars[var][0, -1, level_index].float().cpu().numpy()
