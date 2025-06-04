"""Do a rollout with Aurora to predict the weather."""

import logging
import sys
import time

print("Importing ipex")
from pathlib import Path

import intel_extension_for_pytorch as ipex
import torch
import xarray as xr

from aurora import Batch, Metadata

# Data will be downloaded here.
download_path = Path("../era5/era_v_inf")

static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(
    download_path / "2023-01-01-surface-level.nc", engine="netcdf4"
)
atmos_vars_ds = xr.open_dataset(
    download_path / "2023-01-01-atmospheric.nc", engine="netcdf4"
)


def main(steps):
    start_time_total = time.time()
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
            atmos_levels=tuple(
                int(level) for level in atmos_vars_ds.pressure_level.values
            ),
        ),
    )

    from aurora import Aurora, rollout

    print("loading model")
    model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    model.eval()
    model = model.to("xpu")

    print("doing rollout")
    preds = []
    times = []

    with torch.inference_mode():
        time_start = time.time()
        for pred in rollout(model, batch, steps=steps):
            preds.append(pred.to("cpu"))
            time_end = time.time()
            print(f"Time for one step: {time_end - time_start}")
            times.append(time_end - time_start)
            time_start = time.time()

    avg_time = sum(times[1:]) / len(times[1:])  # Exclude the first step time
    print(f"Average time for last {steps - 1} steps: {avg_time}")
    print(f"Total time for {steps} steps: {sum(times)}")

    import pickle

    end_time_total = time.time()
    print(f"Total time: {end_time_total - start_time_total}")


#    with open("preds.pkl", "wb") as f:
#        pickle.dump(preds, f)


if __name__ == "__main__":
    steps = 2
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])

    main(steps)
