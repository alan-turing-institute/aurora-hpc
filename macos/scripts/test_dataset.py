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
        self.assertEqual(len(dataset), 31)

        dataset = AuroraDataset(
            data_path=Path("../era5/test"),
            t=0,
            static_filepath=Path("small-static.nc"),
            surface_filepath=Path("small-surface-level.nc"),
            atmos_filepath=Path("small-atmospheric.nc"),
        )
        self.assertEqual(len(dataset), 32)

    def test_get_item(self):
        dataset = AuroraDataset(
            data_path=Path("../era5/test"),
            t=1,
            static_filepath=Path("small-static.nc"),
            surface_filepath=Path("small-surface-level.nc"),
            atmos_filepath=Path("small-atmospheric.nc"),
        )
        batch = dataset[0]
        self.assertIn("2t", batch.surf_vars.keys())
        self.assertEqual(
            # (b, t, h, w)
            batch.surf_vars["2t"].shape, (1, 2, 2, 2)  

        )
        
    def test_get_item_2(self):
        dataset = AuroraDataset(
            data_path=Path("../era5/test"),
            t=2,
            static_filepath=Path("small-static.nc"),
            surface_filepath=Path("small-surface-level.nc"),
            atmos_filepath=Path("small-atmospheric.nc"),
        )
        batch = dataset[0]
        self.assertIn("10u", batch.surf_vars.keys())
        self.assertEqual(
            # (b, t, h, w)
            batch.surf_vars["10u"].shape, (1, 3, 2, 2)  

        )

if __name__ == "__main__":
    unittest.main()