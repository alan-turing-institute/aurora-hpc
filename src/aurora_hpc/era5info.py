"""Display details about an ERA5 dataset."""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from aurora_hpc.dataset import AuroraDataset  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
)
parser.add_argument(
    "--static",
    "-t",
    help="leafname of the static variables",
    default="static.nc",
)
parser.add_argument(
    "--surface",
    "-s",
    help="leafname of the surface variables",
    default="2023-01-surface-level.nc",
)
parser.add_argument(
    "--atmos",
    "-a",
    help="leafname of the atmospheric variables",
    default="2023-01-atmospheric.nc",
)
args = parser.parse_args()


def main(download_path: str, static: str, surface: str, atmos: str):
    download_path = Path(download_path)

    print("loading data...")
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_data=Path(static),
        surface_data=Path(surface),
        atmos_data=Path(atmos),
    )

    print("parsing data...")
    input, _ = dataset[0]
    print()

    print("Number of datapoints in the dataset: {}".format(len(dataset)))
    print("Static variables: {}".format(", ".join(input.static_vars.keys())))
    print("Surface variables: {}".format(", ".join(input.surf_vars.keys())))
    print("Atmospheric variables: {}".format(", ".join(input.atmos_vars.keys())))

    print(
        "Atmospheric levels: {}".format(
            ", ".join([str(level) for level in input.metadata.atmos_levels])
        )
    )
    print("Latitudes: {}".format(len(input.metadata.lat)))
    print("Longitudes: {}".format(len(input.metadata.lon)))
    print()

    print("Times:")
    for index, (input, _) in enumerate(dataset):
        for time in input.metadata.time:
            print("Batch {}: {}".format(index, time))


main(args.download_path, args.static, args.surface, args.atmos)
