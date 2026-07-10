import unittest
from datetime import datetime

import torch
from aurora_loss import mae

from aurora import Batch, Metadata

SURFACE_WEIGHTS = {
    "2t": 3.0,
    "msl": 1.5,
    "10u": 0.77,
    "10v": 0.66,
}
ATMOS_WEIGHTS = {
    "z": 2.8,
    "q": 0.78,
    "t": 1.7,
    "u": 0.87,
    "v": 0.6,
}


def constant_batch(
    value: float, height: int = 3, width: int = 4, levels: int = 2
) -> Batch:
    return Batch(
        surf_vars={
            key: torch.full((1, 1, height, width), value) for key in SURFACE_WEIGHTS
        },
        static_vars={
            "z": torch.full((height, width), value),
            "slt": torch.full((height, width), value),
            "lsm": torch.full((height, width), value),
        },
        atmos_vars={
            key: torch.full((1, 1, levels, height, width), value)
            for key in ATMOS_WEIGHTS
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, height),
            lon=torch.linspace(0, 360, width + 1)[:-1],
            time=(datetime(2020, 1, 1),),
            atmos_levels=tuple(range(levels)),
        ),
    )


class AuroraLossTests(unittest.TestCase):
    def test_mae_normalises_by_actual_resolution_and_level_count(self):
        pred = constant_batch(1.0)
        target = constant_batch(0.0)

        expected = (2 / 9) * (
            0.25 * sum(SURFACE_WEIGHTS.values()) + sum(ATMOS_WEIGHTS.values())
        )

        self.assertTrue(torch.isclose(mae(pred, target), torch.tensor(expected)))


if __name__ == "__main__":
    unittest.main()
