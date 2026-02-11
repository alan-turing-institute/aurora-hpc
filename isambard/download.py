#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

import argparse
import sys

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute
from pathlib import Path

import cdsapi

c = cdsapi.Client()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "download_path",
        type=Path,
        default="../../datasets/era5",
        help="The location to save the datasets to.",
    )
    parsed = parser.parse_args()
    download_path = parsed.download_path.absolute()

    if not download_path.is_dir():
        print(f"{download_path=} doesn't exist or isn't a directory")
        exit(1)

    times = ["00:00", "06:00", "12:00", "18:00"]

    for x in [
        (
            "static.nc",
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "land_sea_mask",
                    "soil_type",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
        ),
        (
            "2023-01-01-surface-level.nc",
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": times,
                "format": "netcdf",
            },
        ),
        (
            "2023-01-01-atmospheric.nc",
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ],
                "pressure_level": [
                    "50",
                    "100",
                    "150",
                    "200",
                    "250",
                    "300",
                    "400",
                    "500",
                    "600",
                    "700",
                    "850",
                    "925",
                    "1000",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": times,
                "format": "netcdf",
            },
        ),
    ]:
        the_path = download_path / x[0]
        if not the_path.exists():
            print("Retrieving", x[0])
            c.retrieve(
                x[1],
                x[2],
                the_path,
            )
            print("Retrieved")


if __name__ == "__main__":
    main(sys.argv)
