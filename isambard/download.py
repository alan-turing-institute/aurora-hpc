#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

import argparse

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute
from pathlib import Path

import cdsapi

c = cdsapi.Client()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argumet(
        "download_path",
        type=Path,
        default="../../datasets/era5",
        help="The location to save the datasets to.",
    )
    parsed = parser.parse_args()
    download_path = parsed.download_path.abspath()

    if not download_path.isdir():
        print(f"{download_path=} doesn't exist or isn't a directory")
        exit(1)

    # Download the static variables.
    if not (download_path / "static.nc").exists():
        c.retrieve(
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
            str(download_path / "static.nc"),
        )
    print("Static variables downloaded!")

    # Download the surface-level variables.
    if not (download_path / "2023-01-01-surface-level.nc").exists():
        c.retrieve(
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
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(download_path / "2023-01-01-surface-level.nc"),
        )
    print("Surface-level variables downloaded!")

    # Download the atmospheric variables.
    if not (download_path / "2023-01-01-atmospheric.nc").exists():
        c.retrieve(
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
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(download_path / "2023-01-01-atmospheric.nc"),
        )
    print("Atmospheric variables downloaded!")


if __name__ == "__main__":
    main(sys.argv)
