#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import torch
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import numpy as np
from aurora_loss import mae
from torch.nn import MSELoss

from aurora import Aurora, rollout, Batch, Metadata
from pathlib import Path

print("Loading dataset")
# Data will be downloaded here.
download_path = Path("../era5-experiments/downloads")
download_path = download_path.expanduser()

static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(
    download_path / "2023-01-01-08-surface-level.nc", engine="netcdf4"
)
atmos_vars_ds = xr.open_dataset(
    download_path / "2023-01-01-08-atmospheric.nc", engine="netcdf4"
)

def load_data(filename):
    print("Loading pickle file: {}".format(filename))
    with open(filename, "rb") as f:
        preds = pickle.load(f)
    return preds

def plot_predict_vs_ground(preds, filename):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    step = 27
    pred = preds[step]

    ax[0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[0].set_ylabel(str(pred.metadata.time[0]))
    ax[0].set_title("Aurora Prediction")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(surf_vars_ds["t2m"][2 + step].values - 273.15, vmin=-50, vmax=50)
    ax[1].set_title("ERA5")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def calculate_rmse(preds0, preds1):
    return np.sqrt(np.mean((preds0 - preds1)**2))

def calculate_difference(vars0, vars1):
    return abs(vars0 - vars1)

def plot_error_comparison(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    rmse = []

    step = 27
    vmin = 0
    vmax = 5

    for step in range(1, 28):
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

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(rmse, linestyle="", marker="x")

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Root Mean Square Error")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def plot_errors(preds_dawn, preds_bask, filename):
    print("Plotting graph: {}".format(filename))
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

    step = 27
    vmin = 0
    #vmax = 5

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

    img = ax[0, 0].imshow(diff_pred_actual_dawn, vmin=vmin)
    ax[0, 0].set_ylabel(str(preds_dawn[step].metadata.time[0]))
    ax[0, 0].set_xlabel("RMSE: {:1.5f}".format(rmse_pred_actual_dawn))
    ax[0, 0].set_title("Error on DAWN")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)

    img = ax[0, 1].imshow(diff_pred_actual_bask, vmin=vmin)
    ax[0, 1].set_title("Error on Baskerville")
    ax[0, 1].set_xlabel("RMSE: {:1.5f}".format(rmse_pred_actual_bask))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)

    img = ax[1, 0].imshow(diff_dawn_bask_pred, vmin=vmin)
    ax[1, 0].set_ylabel(str(preds_bask[step].metadata.time[0]))
    ax[1, 0].set_title("DAWN vs. Baskerville on Predictions")
    ax[1, 0].set_xlabel("RMSE: {:1.5f}".format(rmse_dawn_bask_pred))
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)

    img = ax[1, 1].imshow(diff_dawn_bask_actual, vmin=vmin)
    ax[1, 1].set_title("DAWN vs. Baskerville on Actual")
    ax[1, 1].set_xlabel("RMSE: {:1.5f}".format(rmse_dawn_bask_actual))
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    c_bar = plt.colorbar(img, orientation="vertical", pad=0.05, shrink=0.73)


    #plt.tight_layout()
    #fig.suptitle("Absolute error comparison for two-meter temperature in K ranged (0, 5) at rollout step 28")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, )

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
            loss_list.append(losses)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(loss_list[0], linestyle="", marker="x", label="DAWN")
    ax.plot(loss_list[1], linestyle="", marker="+", label="Baskerville")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean Average Error")
    ax.legend();

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

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
    plt.savefig(filename, dpi=300)

preds_dawn = load_data("preds-dawn.pkl")
preds_bask = load_data("preds-bask.pkl")

plot_predict_vs_ground(preds_dawn, "plot-pvg-dawn.pdf")
plot_predict_vs_ground(preds_bask, "plot-pvg-bask.pdf")
plot_predict_vs_ground(preds_dawn, "plot-pvg-dawn.png")
plot_predict_vs_ground(preds_bask, "plot-pvg-bask.png")
plot_errors(preds_dawn, preds_bask, "plot-errors.pdf")
plot_errors(preds_dawn, preds_bask, "plot-errors.png")
plot_error_comparison(preds_dawn, preds_bask, "plot-error-comparison.pdf")
plot_error_comparison(preds_dawn, preds_bask, "plot-error-comparison.png")
plot_losses(preds_dawn, preds_bask, "plot-losses.pdf")
plot_losses(preds_dawn, preds_bask, "plot-losses.png")
plot_var_losses(preds_dawn, preds_bask, "plot-var-losses.pdf")
plot_var_losses(preds_dawn, preds_bask, "plot-var-losses.png")
