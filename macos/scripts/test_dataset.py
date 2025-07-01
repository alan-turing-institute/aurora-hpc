import unittest
import itertools
import datetime
from dataset import AuroraDataset
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np

def make_dummy_static(temp_directory: Path) -> str:
    """Make a dummy static dataset and return the file name."""
    dims = {'valid_time': 1, 'latitude': 2, 'longitude': 2}

    # Optionally, create some coordinate values
    now = datetime.datetime.now()
    coords = {
        'valid_time': np.arange(now, now + datetime.timedelta(seconds=dims['valid_time']), dtype="datetime64[s]"),
        'latitude': np.linspace(-90, 90, dims['latitude'], dtype=np.float64),
        'longitude': np.linspace(-180, 180, dims['longitude'], dtype=np.float64),
    }

    # Make dummy data (e.g., all zeros)
    data = np.zeros((dims['valid_time'], dims['latitude'], dims['longitude']), dtype=np.float32)

    # Wrap in a DataArray
    array = xr.DataArray(
        data,
        dims=('valid_time', 'latitude', 'longitude'),
        coords=coords,
        # name='dummy_variable'
    )

    dataset = xr.Dataset({x: array for x in ("z", "lsm", "slt")})
    file_name = "test-static.nc"
    dataset.to_netcdf(temp_directory / file_name, engine="netcdf4")
    return file_name

def make_dummy_surface(temp_directory: Path) -> str:
    """Make a dummy surface dataset and return the path to it."""
    dims = {'valid_time': 32, 'latitude': 2, 'longitude': 2}

    # Optionally, create some coordinate values
    now = datetime.datetime.now()
    coords = {
        'valid_time': np.arange(now, now + datetime.timedelta(seconds=dims['valid_time']), dtype="datetime64[s]"),
        'latitude': np.linspace(-90, 90, dims['latitude'], dtype=np.float64),
        'longitude': np.linspace(-180, 180, dims['longitude'], dtype=np.float64),
    }

    # Make dummy data (e.g., all zeros)
    data = np.zeros((dims['valid_time'], dims['latitude'], dims['longitude']), dtype=np.float32)

    # Wrap in a DataArray
    array = xr.DataArray(
        data,
        dims=('valid_time', 'latitude', 'longitude'),
        coords=coords,
        # name='dummy_variable'
    )

    dataset = xr.Dataset({x: array for x in ("t2m", "u10", "v10", "msl")})
    file_name = "test-surface.nc"
    dataset.to_netcdf(temp_directory / file_name, engine="netcdf4")
    return file_name

def make_dummy_atmos(temp_directory: Path) -> str:
    """Make a dummy atmospheric dataset and return the file name."""
    dims = {'valid_time': 32, 'pressure_level': 13, 'latitude': 2, 'longitude': 2}

    # Optionally, create some coordinate values
    now = datetime.datetime.now()
    coords = {
        'valid_time': np.arange(now, now + datetime.timedelta(seconds=dims['valid_time']), dtype="datetime64[s]"),
        'pressure_level': np.linspace(0, 999, dims['pressure_level'], dtype=np.float64),
        'latitude': np.linspace(-90, 90, dims['latitude'], dtype=np.float64),
        'longitude': np.linspace(-180, 180, dims['longitude'], dtype=np.float64),
    }

    # Make dummy data (e.g., all zeros)
    data = np.zeros((dims['valid_time'], dims['pressure_level'], dims['latitude'], dims['longitude']), dtype=np.float32)

    # Wrap in a DataArray
    array = xr.DataArray(
        data,
        dims=('valid_time', 'pressure_level', 'latitude', 'longitude'),
        coords=coords,
        # name='dummy_variable'
    )

    dataset = xr.Dataset({x: array for x in ("t", "u", "v", "q", "z")})
    file_name = "test-atmos.nc"
    dataset.to_netcdf(temp_directory / file_name, engine="netcdf4")
    return file_name


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Should be deleted when the variable goes out of scope.
        cls.temp_dir = TemporaryDirectory()
        print(f"Temporary directory created at {cls.temp_dir.name}")
        cls.dummy_static = make_dummy_static(Path(cls.temp_dir.name))
        cls.dummy_surface = make_dummy_surface(Path(cls.temp_dir.name))
        cls.dummy_atmos = make_dummy_atmos(Path(cls.temp_dir.name))


    def test_dataset_length(self):
        dataset = AuroraDataset(
            data_path=Path(self.temp_dir.name),
            t=1,
            static_filepath=self.dummy_static,
            surface_filepath=self.dummy_surface,
            atmos_filepath=self.dummy_atmos,
        )
        self.assertEqual(len(dataset), 30)

        dataset = AuroraDataset(
            data_path=Path("../era5/test"),
            t=0,
            static_filepath=Path("small-static.nc"),
            surface_filepath=Path("small-surface-level.nc"),
            atmos_filepath=Path("small-atmospheric.nc"),
        )
        self.assertEqual(len(dataset), 31)

    def test_get_item(self):
        for (in_t, out_t), var_name in itertools.product([(1, 2), (2, 3)], ["2t", "10u", "10v", "msl"]):
            with self.subTest(in_t=in_t, out_t=out_t, var_name=var_name):
                dataset = AuroraDataset(
                    data_path=Path("../era5/test"),
                    t=in_t,
                    static_filepath=Path("small-static.nc"),
                    surface_filepath=Path("small-surface-level.nc"),
                    atmos_filepath=Path("small-atmospheric.nc"),
                )
                X, y = dataset[0]
                self.assertEqual(
                    # (b, t, h, w)
                    X.surf_vars[var_name].shape, (1, out_t, 2, 2)  

                )
                self.assertEqual(
                    y.surf_vars[var_name].shape, (1, 2, 2)  
                )
        

if __name__ == "__main__":
    unittest.main()