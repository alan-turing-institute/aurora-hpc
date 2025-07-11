#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import argparse
import logging
import sys
import time

from pathlib import Path

import torch

from aurora_hpc.dataset import AuroraDataset


def main(
    download_path: str, nsteps: int, save: bool = False, output_file: str = "preds.pkl"
):
    time_start_total = time.time()

    print("Loading data...")
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
    )  # Defaults to the 2023-01-01 dataset
    print("Getting input batch...")
    batch = dataset[0][0]  # Get the first item in the dataset.

    from aurora import Aurora, rollout

    print("loading model")
    model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    model.eval()
    model = model.to("cuda")

    print("doing rollout")
    preds = []
    times = []

    with torch.inference_mode():
        time_start = time.time()
        for pred in rollout(model, batch, steps=nsteps):
            preds.append(pred.to("cpu"))
            time_end = time.time()
            print(f"Time for one step: {time_end - time_start}")
            times.append(time_end - time_start)
            time_start = time.time()

    avg_time = sum(times[1:]) / len(times[1:])  # Exclude the first step time
    print(f"Average time for last {nsteps - 1} steps: {avg_time}")
    print(f"Total time for {nsteps} steps: {sum(times)}")

    import pickle

    time_end_total = time.time()
    print(f"Total time: {time_end_total - time_start_total}")

    if save:
        print(f"Saving predictions to {output_file}")
        with open(output_file, "wb") as f:
            pickle.dump(preds, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_path",
        "-d",
        help="path to download directory",
        default="../era5-experiments/downloads",
    )
    parser.add_argument(
        "--nsteps",
        "-n",
        type=int,
        help="number of steps to roll out",
        default=2,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="whether to save the predictions",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="name of file to save outputs (should be a .pkl)",
        default="preds.pkl",
    )
    args = parser.parse_args()

    main(
        args.download_path,
        nsteps=args.nsteps,
        save=args.save,
        output_file=args.output_file,
    )
