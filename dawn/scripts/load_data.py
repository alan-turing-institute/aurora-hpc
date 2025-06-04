from pathlib import Path

import xarray as xr


def load_data(input_path: str = "../era5/era_v_inf"):
    """Load the ERA5 data"""
    download_path = Path(input_path)

    print("Loading data..")
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(
        download_path / "2023-01-surface-level.nc", engine="netcdf4"
    )
    atmos_vars_ds = xr.open_dataset(
        download_path / "2023-01-atmospheric.nc", engine="netcdf4"
    )
    return static_vars_ds, surf_vars_ds, atmos_vars_ds
