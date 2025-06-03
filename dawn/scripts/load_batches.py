import torch

from aurora import Batch, Metadata


def get_input_batch(i: int, static_vars_ds, surf_vars_ds, atmos_vars_ds):
    """Get batch for i-th time index"""
    print(f"Getting batch for i={i} and i-1")
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
            atmos_levels=tuple(
                int(level) for level in atmos_vars_ds.pressure_level.values
            ),
        ),
    )
    return batch


def get_gt_batch(i: int, static_vars_ds, surf_vars_ds, atmos_vars_ds):
    """Get i+1 data, i.e. prediction from i-th time index"""
    print(f"Getting ground truth results for i={i}")
    batch = Batch(
        surf_vars={
            # First select time point `i + 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i + 1]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i + 1]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i + 1]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i + 1]][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i + 1]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i + 1]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i + 1]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i + 1]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i + 1]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(
                surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i + 1],
            ),
            atmos_levels=tuple(
                int(level) for level in atmos_vars_ds.pressure_level.values
            ),
        ),
    )
    return batch
