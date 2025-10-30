#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute
from pathlib import Path

import argparse
import torch
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torch.nn import MSELoss

from aurora import Aurora, rollout, Batch, Metadata
from aurora_hpc.aurora_loss import mae
from aurora_hpc.dataset import batch_collate_fn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
)
parser.add_argument(
    "--image_type",
    "-i",
    help="image type to plot (as a file extension)",
    default="pdf",
)
parser.add_argument(
    "--num_files",
    "-n",
    type=int,
    help="The number of input data files to read for averaging",
    default=4,
)
args = parser.parse_args()

print("Output format: {}".format(args.image_type))
print("Number of input files to read: {}".format(args.num_files))

assert args.num_files > 0

SURF_VARS_DS_KEYS_MAP = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
}

print("Loading dataset")
# Data will be downloaded here.
download_path = Path(args.download_path)
download_path = download_path.expanduser()

static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(
    download_path / "2023-01-surface-level.nc", engine="netcdf4"
)
atmos_vars_ds = xr.open_dataset(
    download_path / "2023-01-atmospheric.nc", engine="netcdf4"
)

weatherbench2_ifs_ens_mean_2m = [
    0.7046725,
    0.6250805,
    0.7371223,
    0.7870529,
    0.8339144,
    0.8677869,
    0.9096333,
    0.938898,
    0.9709085,
    0.998581,
    1.033265,
    1.063641,
    1.095916,
    1.128267,
    1.164034,
    1.20062,
    1.236261,
    1.275574,
    1.314509,
    1.357237,
    1.3961,
    1.441059,
    1.481914,
    1.528329,
    1.567438,
    1.615482,
    1.653148,
    1.699029,
]

weatherbench2_ifs_ens_first_2m = [
    0.860784,
    0.794231,
    0.9006215,
    0.960666,
    1.013409,
    1.063852,
    1.114111,
    1.160714,
    1.20238,
    1.250151,
    1.296216,
    1.348182,
    1.392303,
    1.448409,
    1.498825,
    1.559912,
    1.612841,
    1.676311,
    1.732587,
    1.801894,
    1.858862,
    1.928962,
    1.987371,
    2.061119,
    2.116542,
    2.190844,
    2.246718,
    2.318011,
]

def load_data(filename):
    print("Loading pickle file: {}".format(filename))
    with open(filename, "rb") as f:
        preds = pickle.load(f)
    return preds

def savefig(plt, filename):
    fullname = "{}.{}".format(filename, args.image_type)
    plt.savefig(fullname, dpi=300)

def average_data(
    preds_list: list,
    return_std_devs: bool = False,
):
    """Average data across multiple lists of predictions.

    Parameters
    ----------
    preds_list : list
        List of lists of predictions
    return_std_devs : bool, optional
        Whether to return the standard deviations, by default False

    Returns
    -------
    list
        If `return_std_devs` is False, a list of averaged predictions where each item is a Batch.
        If `return_std_devs` is True, also returns the standard deviations as a list of Batches.
    """
    print("Averaging data across {} predictions".format(len(preds_list)))

    nsteps = len(preds_list[0]) # Number of timesteps predicted

    avg_preds = []
    if return_std_devs:
        std_devs = []

    for step in range(nsteps):
        # Use batch_collate_fn to combine the predictions for this step
        avg_batch = batch_collate_fn(
            [preds[step] for preds in preds_list],
        )
        if return_std_devs:
            from copy import deepcopy
            std_dev = deepcopy(avg_batch)
        # surface vars
        for k, v in avg_batch.surf_vars.items():
            avg_batch.surf_vars[k] = v.mean(dim=0, keepdim=True)
            if return_std_devs:
                std_dev.surf_vars[k] = v.std(dim=0, keepdim=True)
        # atmos vars
        for k, v in avg_batch.atmos_vars.items():
            avg_batch.atmos_vars[k] = v.mean(dim=0, keepdim=True)
            if return_std_devs:
                std_dev.atmos_vars[k] = v.std(dim=0, keepdim=True)

        # static vars can be ignored as they are constant
        # append the averaged batch to the list
        avg_preds.append(avg_batch)
        if return_std_devs:
            std_devs.append(std_dev)
    return_vals = [avg_preds]
    if return_std_devs:
        return_vals.append(std_devs)
    return return_vals if len(return_vals) > 1 else return_vals[0]

def plot_predict_vs_ground(preds, filename, vars_key="2t"):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    step = len(preds) - 1
    pred = preds[step] # use last step

    # to fix e.g. "2t" to "t2m" for surf_vars_ds
    if vars_key not in pred.surf_vars:
        raise ValueError(f"Variable '{vars_key}' not found in prediction surf_vars. Available keys: {list(pred.surf_vars.keys())}")
    ds_key = SURF_VARS_DS_KEYS_MAP.get(vars_key, vars_key)

    data = pred.surf_vars[vars_key][0, 0].numpy()
    gt = surf_vars_ds[ds_key][2 + step].values

    vmin = min(data.min(), gt.min())
    vmax = max(data.max(), gt.max())
    print(f"vmin: {vmin}, vmax: {vmax}")

    ax[0].imshow(data, vmin=vmin, vmax=vmax)
    ax[0].set_ylabel(str(pred.metadata.time[0]))
    ax[0].set_title("Aurora Prediction")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(gt, vmin=vmin, vmax=vmax)
    ax[1].set_title("ERA5")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.tight_layout()
    savefig(plt, filename)

def plot_std_dev_comparison(
    std_devs_dawn: list,
    std_devs_bask: list,
    filename,
    vars_key="2t"
):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    step = min(len(std_devs_dawn), len(std_devs_bask)) - 1 # use last step
    std_devs_dawn = std_devs_dawn[step]
    std_devs_bask = std_devs_bask[step]

    # to fix e.g. "2t" to "t2m" for surf_vars_ds
    if vars_key not in std_devs_dawn.surf_vars:
        raise ValueError(f"Variable '{vars_key}' not found in std_devs_dawn surf_vars. Available keys: {list(std_devs_dawn.surf_vars.keys())}")


    data_dawn = std_devs_dawn.surf_vars[vars_key][0, 0].numpy()
    data_bask = std_devs_bask.surf_vars[vars_key][0, 0].numpy()

    vmin = min(data_dawn.min(), data_bask.min())
    vmax = max(data_dawn.max(), data_bask.max())

    ax[0].imshow(data_dawn, vmin=vmin, vmax=vmax)
    ax[0].set_ylabel(str(std_devs_dawn.metadata.time[0]))
    ax[0].set_title("DAWN Aurora Prediction Std Dev")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(data_bask, vmin=vmin, vmax=vmax)
    ax[1].set_ylabel(str(std_devs_bask.metadata.time[0]))
    ax[1].set_title("Baskerville Aurora Prediction Std Dev")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.tight_layout()
    savefig(plt, filename)

def calculate_rmse(preds0, preds1):
    return np.sqrt(np.mean((preds0 - preds1)**2))

def calculate_difference(vars0, vars1):
    return abs(vars0 - vars1)

def plot_error_comparison(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    rmse = []

    steps = min(len(preds_dawn), len(preds_bask)) - 1 # use last step

    for step in range(1, steps):
        vars_preds_dawn = preds_dawn[step].surf_vars["2t"][0, 0].numpy()
        vars_preds_bask = preds_bask[step].surf_vars["2t"][0, 0].numpy()

        rmse_dawn_bask_pred = calculate_rmse(
            vars_preds_dawn,
            vars_preds_bask,
        )
        rmse.append(rmse_dawn_bask_pred)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(rmse, linestyle="", marker="x")

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Root Mean Square Error")

    plt.tight_layout()
    savefig(plt, filename)

def plot_weatherbench_comparison(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    rmse = []

    step = 27
    vmin = 0
    vmax = 5
    steps = range(1, 28)

    for step in steps:
        vars_preds_dawn = preds_dawn[step].surf_vars["2t"][0, 0].numpy()
        vars_preds_bask = preds_bask[step].surf_vars["2t"][0, 0].numpy()
        vars_actual = surf_vars_ds["t2m"][2 + step][0:720,:].values

        diff_dawn_bask_pred = calculate_difference(
            vars_preds_dawn,
            vars_preds_bask,
        )
        rmse_dawn_bask_pred = calculate_rmse(
            vars_preds_dawn,
            vars_preds_bask,
        )
        rmse.append(rmse_dawn_bask_pred)
        #print("DB step {}, error: {}".format(step, rmse_dawn_bask_pred))
        #print("IFS mean step {}, error: {}".format(step, weatherbench2_ifs_ens_mean_2m[step]))
        #print("IFS first step {}, error: {}".format(step, weatherbench2_ifs_ens_first_2m[step]))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(steps, rmse, linestyle="-", marker="x", color="#9cd839", label="RMSE between DAWN and Baskerville")

    ax.plot(steps, weatherbench2_ifs_ens_mean_2m[1:28], linestyle="-", marker="", color="#4dc169", label="IFS ENS (mean) vs Analysis")
    ax.errorbar(steps, weatherbench2_ifs_ens_mean_2m[1:28], yerr=rmse, color="#4dc169")

    ax.plot(steps, weatherbench2_ifs_ens_first_2m[1:28], linestyle="-", marker="", color="#228b8b", label="IFS ENS (1st member) vs Analysis")
    ax.errorbar(steps, weatherbench2_ifs_ens_first_2m[1:28], yerr=rmse, color="#228b8b")

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Root Mean Square Error")

    plt.legend()

    plt.tight_layout()
    savefig(plt, filename)

def plot_errors(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

    step = min(len(preds_dawn), len(preds_bask)) - 1 # use last step
    vmin = 0

    vars_preds_dawn = preds_dawn[step].surf_vars["2t"][0, 0].numpy()
    vars_preds_bask = preds_bask[step].surf_vars["2t"][0, 0].numpy()
    vars_actual = surf_vars_ds["t2m"][2 + step][0:720,:].values

    diff_pred_actual_dawn = calculate_difference(
        vars_preds_dawn,
        vars_actual,
    )
    rmse_pred_actual_dawn = calculate_rmse(
        preds_dawn[step].surf_vars["2t"][0, 0].numpy(),
        vars_actual,
    )
    print("RMSE prediction vs. actual on DAWN (step {}): {}".format(step, rmse_pred_actual_dawn))

    diff_pred_actual_bask = calculate_difference(
        vars_preds_bask,
        vars_actual,
    )
    rmse_pred_actual_bask = calculate_rmse(
        vars_preds_bask,
        vars_actual,
    )
    print("RMSE prediction vs. actual on Baskerville (step {}): {}".format(step, rmse_pred_actual_bask))

    diff_dawn_bask_pred = calculate_difference(
        vars_preds_dawn,
        vars_preds_bask,
    )
    rmse_dawn_bask_pred = calculate_rmse(
        vars_preds_dawn,
        vars_preds_bask,
    )
    print("RMSE DAWN vs. Baskerville on Predictions (step {}): {}".format(step, rmse_dawn_bask_pred))

    diff_dawn_bask_actual = calculate_difference(
        vars_actual,
        vars_actual,
    )
    rmse_dawn_bask_actual = calculate_rmse(
        vars_actual,
        vars_actual,
    )
    print("RMSE DAWN vs. Baskerville on Actual (step {}): {}".format(step, rmse_dawn_bask_actual))

    vmax = max(
        diff_pred_actual_dawn.max(),
        diff_pred_actual_bask.max(),
    )

    img = ax[0, 0].imshow(diff_pred_actual_dawn, vmin=vmin, vmax=vmax)
    ax[0, 0].set_ylabel(str(preds_dawn[step].metadata.time[0]))
    ax[0, 0].set_xlabel("RMSE: {:1.5f}".format(rmse_pred_actual_dawn))
    ax[0, 0].set_title("Error on DAWN")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)

    img = ax[0, 1].imshow(diff_pred_actual_bask, vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("Error on Baskerville")
    ax[0, 1].set_xlabel("RMSE: {:1.5f}".format(rmse_pred_actual_bask))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)

    vmax = max(
        diff_dawn_bask_pred.max(),
        diff_dawn_bask_actual.max(),
    )

    img = ax[1, 0].imshow(diff_dawn_bask_pred, vmin=vmin, vmax=vmax)
    ax[1, 0].set_ylabel(str(preds_bask[step].metadata.time[0]))
    ax[1, 0].set_title("DAWN vs. Baskerville on Predictions")
    ax[1, 0].set_xlabel("RMSE: {:1.5f}".format(rmse_dawn_bask_pred))
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)

    img = ax[1, 1].imshow(diff_dawn_bask_actual, vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("DAWN vs. Baskerville on Actual")
    ax[1, 1].set_xlabel("RMSE: {:1.5f}".format(rmse_dawn_bask_actual))
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)


    #plt.tight_layout()
    #fig.suptitle("Absolute error comparison for two-meter temperature in K ranged (0, 5) at rollout step 28")
    plt.tight_layout()
    savefig(plt, filename)

def plot_losses(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    loss_list = []
    for preds in [preds_dawn, preds_bask]:
        losses = []
        for i, pred in enumerate(preds):
            batch = Batch(
                surf_vars={
                    # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
                    # batch dimension of size one.
                    "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i+2]][None]),
                    "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i+2]][None]),
                    "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i+2]][None]),
                    "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i+2]][None]),
                },
                static_vars={
                    # The static variables are constant, so we just get them for the first time.
                    "z": torch.from_numpy(static_vars_ds["z"].values[0]),
                    "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
                    "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
                },
                atmos_vars={
                    "t": torch.from_numpy(atmos_vars_ds["t"].values[[i+2]][None]),
                    "u": torch.from_numpy(atmos_vars_ds["u"].values[[i+2]][None]),
                    "v": torch.from_numpy(atmos_vars_ds["v"].values[[i+2]][None]),
                    "q": torch.from_numpy(atmos_vars_ds["q"].values[[i+2]][None]),
                    "z": torch.from_numpy(atmos_vars_ds["z"].values[[i+2]][None]),
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
            loss = mae(pred, batch)
            losses.append(loss.item())
        # [0] for dawn, [1] for baskerville
        loss_list.append(losses)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(loss_list[0], linestyle="", marker="x", label="DAWN")
    ax.plot(loss_list[1], linestyle="", marker="+", label="Baskerville")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean Average Error")
    ax.legend()

    plt.tight_layout()
    savefig(plt, filename)

def plot_var_losses(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    surf_losses_list = []
    atmos_losses_list = []
    for preds in [preds_dawn, preds_bask]:
        surf_losses = {}
        atmos_losses = {}

        for i, pred in enumerate(preds):
            batch = Batch(
                surf_vars={
                    # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
                    # batch dimension of size one.
                    "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i+2]][None]),
                    "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i+2]][None]),
                    "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i+2]][None]),
                    "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i+2]][None]),
                },
                static_vars={
                    # The static variables are constant, so we just get them for the first time.
                    "z": torch.from_numpy(static_vars_ds["z"].values[0]),
                    "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
                    "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
                },
                atmos_vars={
                    "t": torch.from_numpy(atmos_vars_ds["t"].values[[i+2]][None]),
                    "u": torch.from_numpy(atmos_vars_ds["u"].values[[i+2]][None]),
                    "v": torch.from_numpy(atmos_vars_ds["v"].values[[i+2]][None]),
                    "q": torch.from_numpy(atmos_vars_ds["q"].values[[i+2]][None]),
                    "z": torch.from_numpy(atmos_vars_ds["z"].values[[i+2]][None]),
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

            loss_fn = MSELoss()

            for k, v in pred.surf_vars.items():
                loss = loss_fn(v, batch.surf_vars[k][:, :, :720, :]).item()
                if k not in surf_losses:
                    surf_losses[k] = [loss]
                else:
                    surf_losses[k].append(loss)

            for k, v in pred.atmos_vars.items():
                loss = loss_fn(v, batch.atmos_vars[k][:, :, :, :720, :]).item()
                if k not in atmos_losses:
                    atmos_losses[k] = [loss]
                else:
                    atmos_losses[k].append(loss)

        surf_losses_list.append(surf_losses)
        atmos_losses_list.append(atmos_losses)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,10))

    labels = ["DAWN", "Baskerville"]
    markers = ["x", "+"]
    for index in range(len(surf_losses_list)):
        for i, k in enumerate(surf_losses_list[index]):
            i_0 = i//2
            i_1 = i%2
            axs[i_0, i_1].plot(
                surf_losses_list[index][k],
                linestyle="",
                marker=markers[index],
                label=labels[index]
            )
            axs[i_0, i_1].set_title("Variable {}".format(k))
            axs[i_0, i_1].legend()
    axs[1, 0].set_xlabel("Rollout step")
    axs[1, 1].set_xlabel("Rollout step")

    axs[0, 0].set_ylabel("Mean Average Error")
    axs[1, 0].set_ylabel("Mean Average Error")

    plt.tight_layout()
    savefig(plt, filename)

preds_dawn = [load_data(f"preds_{i}-dawn.pkl") for i in range(args.num_files)]
preds_bask = [load_data(f"preds_{i}-bask.pkl") for i in range(args.num_files)]

if args.num_files == 1:
    avg_preds_dawn = preds_dawn[0]
    avg_preds_bask = preds_bask[0]
else:
    avg_preds_dawn = average_data(preds_dawn)
    avg_preds_bask = average_data(preds_bask)

# Generate plots
plot_predict_vs_ground(avg_preds_dawn, "plot-pvg-dawn")
plot_predict_vs_ground(avg_preds_bask, "plot-pvg-bask")
plot_errors(avg_preds_dawn, avg_preds_bask, "plot-errors")
plot_error_comparison(avg_preds_dawn, avg_preds_bask, "plot-error-comparison")
plot_losses(avg_preds_dawn, avg_preds_bask, "plot-losses")
plot_var_losses(avg_preds_dawn, avg_preds_bask, "plot-var-losses")
plot_weatherbench_comparison(avg_preds_dawn, avg_preds_bask, "plot-weatherbench-comparison")

if args.num_files > 1:
    # Plot reproducibility comparison
    # This plot is only valid if we have a range of results
    avg_preds_dawn, std_devs_dawn = average_data(preds_dawn, return_std_devs=True)
    avg_preds_bask, std_devs_bask = average_data(preds_bask, return_std_devs=True)
    plot_std_dev_comparison(std_devs_dawn, std_devs_bask, "plot-std-dev-comparison")

