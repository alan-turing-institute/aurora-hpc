import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from dataset import AuroraDataset

SURFACE_VARS = ("tas", "uas", "vas", "psl")
ATMOS_VARS = ("ta", "ua", "va", "hus", "zg")
STANDARD_GRAVITY = 9.80665


def write_build_dataset(root: Path) -> None:
    build = root / "build"
    times = np.array(
        [
            "1850-01-01T06:00:00",
            "1850-01-01T12:00:00",
            "1850-01-01T18:00:00",
            "1850-01-02T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    lat = np.array([90.0, 89.75, 89.5])
    lon = np.array([0.0, 0.25, 0.5, 0.75])
    plev = np.array([5000, 10000])

    for idx, variable in enumerate(SURFACE_VARS):
        variable_dir = build / "surface.regridded" / variable
        variable_dir.mkdir(parents=True)
        data = np.full((len(times), len(lat), len(lon)), idx, dtype=np.float32)
        xr.Dataset(
            {variable: (("time", "lat", "lon"), data)},
            coords={"time": times, "lat": lat, "lon": lon},
        ).to_netcdf(
            variable_dir / f"{variable}_185001010600_185001020000.0p25deg.regridded.nc"
        )

    for idx, variable in enumerate(ATMOS_VARS):
        variable_dir = build / "atmos.regridded" / variable
        variable_dir.mkdir(parents=True)
        value = 2.0 if variable == "zg" else float(idx)
        data = np.full(
            (len(times), len(plev), len(lat), len(lon)),
            value,
            dtype=np.float32,
        )
        attrs = (
            {"units": "m", "standard_name": "geopotential_height"}
            if variable == "zg"
            else {}
        )
        xr.Dataset(
            {variable: (("time", "plev", "lat", "lon"), data, attrs)},
            coords={"time": times, "plev": plev, "lat": lat, "lon": lon},
        ).to_netcdf(
            variable_dir / f"{variable}_185001010600_185001020000.0p25deg.regridded.nc"
        )

    xr.Dataset(
        {
            "z": (
                ("latitude", "longitude"),
                np.zeros((len(lat), len(lon)), dtype=np.float32),
            ),
            "slt": (
                ("latitude", "longitude"),
                np.ones((len(lat), len(lon)), dtype=np.float32),
            ),
            "lsm": (
                ("latitude", "longitude"),
                np.ones((len(lat), len(lon)), dtype=np.float32),
            ),
        },
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(root / "0pt25_static.nc")


class AuroraDatasetBuildTests(unittest.TestCase):
    def test_loads_workflow_build_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_build_dataset(data_path)

            dataset = AuroraDataset(data_path, t=1)

            self.assertEqual(len(dataset), 2)
            inputs, target = dataset[0]

            self.assertEqual(inputs.surf_vars["2t"].shape, (1, 2, 3, 4))
            self.assertEqual(target.surf_vars["2t"].shape, (1, 1, 3, 4))
            self.assertEqual(inputs.atmos_vars["t"].shape, (1, 2, 2, 3, 4))
            self.assertEqual(inputs.static_vars["z"].shape, (3, 4))
            self.assertEqual(inputs.metadata.atmos_levels, (50, 100))
            self.assertEqual(inputs.metadata.time, (datetime(1850, 1, 1, 12),))

    def test_converts_zg_geopotential_height_to_geopotential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_build_dataset(data_path)

            dataset = AuroraDataset(data_path, t=1)
            inputs, target = dataset[0]

            np.testing.assert_allclose(
                inputs.atmos_vars["z"].numpy(),
                2.0 * STANDARD_GRAVITY,
            )
            np.testing.assert_allclose(
                target.atmos_vars["z"].numpy(),
                2.0 * STANDARD_GRAVITY,
            )


if __name__ == "__main__":
    unittest.main()
