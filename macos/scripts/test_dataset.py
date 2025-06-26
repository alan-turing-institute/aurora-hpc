import unittest
from dataset import AuroraDataset
from pathlib import Path

class TestDataset(unittest.TestCase):
    def test_dataset_length(self):
        dataset = AuroraDataset(
            data_path=Path("../era5/test"),
            t=1,
            static_filepath=Path("small-static.nc"),
            surface_filepath=Path("small-surface-level.nc"),
            atmos_filepath=Path("small-atmospheric.nc"),
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
        for in_t, out_t in [(1, 2), (2, 3)]:
            for var_name in ["2t", "10u", "10v", "msl"]:
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