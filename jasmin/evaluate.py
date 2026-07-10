"""Evaluate base vs fine-tuned Aurora on the regridded CMIP6 data.

For each initialisation time, every model produces an autoregressive rollout
(6-hourly steps) which is verified against the held-out CMIP6 fields with
latitude-weighted RMSE/MAE/bias per variable (and per pressure level for the
atmospheric variables). Persistence and monthly-climatology baselines anchor
the comparison.

Outputs, written to --out:
  metrics.csv            long-format scores: model, init_time, lead_hours, var, level, ...
  fields_<model>.nc      selected 2D fields per lead time for one init (--save_fields_init),
                         for maps and case studies. "truth" is written as its own model.

Typical GPU usage (from the repo root):
  python evaluate.py --checkpoint checkpoints/gpu/epoch-0004.pt --device cuda
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from inference import (
    LazyCMIP6Data,
    atmos_field,
    compute_metrics,
    load_finetuned,
    surf_field,
)

from aurora import AuroraPretrained, AuroraSmallPretrained, Batch, rollout

LOGGER = logging.getLogger(__name__)

SAVE_SURF = ("2t", "msl")
SAVE_ATMOS = (("z", 500), ("t", 850))


def build_model(name: str, checkpoint: Path | None, small: bool) -> torch.nn.Module:
    model = AuroraSmallPretrained() if small else AuroraPretrained()
    model.load_checkpoint()
    if name == "finetuned":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for the finetuned model")
        load_finetuned(model, checkpoint)
    model.eval()
    return model


def climatology_batch(data: LazyCMIP6Data) -> Batch:
    """Batch holding the time-mean of every variable over the whole dataset."""
    LOGGER.info("Computing climatology (time mean over %s steps)...", len(data))
    surf_mean = data.surf_ds.mean("valid_time").compute()
    atmos_mean = data.atmos_ds.mean("valid_time").compute()
    template = data.truth_at(0)
    for key, ds_name in {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}.items():
        template.surf_vars[key] = torch.from_numpy(surf_mean[ds_name].values)[
            None, None
        ]
    for key in ("t", "u", "v", "q", "z"):
        template.atmos_vars[key] = torch.from_numpy(atmos_mean[key].values)[None, None]
    return template


def save_field_records(field_records: dict, lead_hours: int, batch: Batch) -> None:
    fields = {var: surf_field(batch, var) for var in SAVE_SURF}
    fields.update(
        {f"{var}{level}": atmos_field(batch, var, level) for var, level in SAVE_ATMOS}
    )
    field_records.setdefault("lead_hours", []).append(lead_hours)
    for name, values in fields.items():
        field_records.setdefault(name, []).append(values)


def write_fields(
    out_dir: Path, model: str, field_records: dict, data: LazyCMIP6Data, init_time
) -> None:
    lead = np.array(field_records.pop("lead_hours"), dtype=np.int64)
    sample = next(iter(field_records.values()))[0]
    lat = data.lat.numpy()[: sample.shape[0]]
    lon = data.lon.numpy()[: sample.shape[1]]
    ds = xr.Dataset(
        {
            name: (("lead_hours", "lat", "lon"), np.stack(values).astype(np.float32))
            for name, values in field_records.items()
        },
        coords={"lead_hours": lead, "lat": lat, "lon": lon},
        attrs={"init_time": str(init_time), "model": model},
    )
    path = out_dir / f"fields_{model}.nc"
    ds.to_netcdf(path)
    LOGGER.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=Path("data"), type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="fine-tuned train.py checkpoint (.pt)",
    )
    parser.add_argument(
        "--models",
        default="base,finetuned,persistence",
        help="comma-separated subset of base,finetuned,persistence,climatology",
    )
    parser.add_argument(
        "--inits",
        default="0,16,32,48,64,80",
        help="comma-separated dataset indices of the first input timestep",
    )
    parser.add_argument(
        "--steps", default=20, type=int, help="rollout length (6h steps)"
    )
    parser.add_argument(
        "--history_t",
        default=1,
        type=int,
        help="additional input timesteps, as in train.py (t=1)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default=Path("eval_results"), type=Path)
    parser.add_argument(
        "--save_fields_init",
        type=int,
        default=None,
        help="init index for which to save 2D fields (default: first init)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="use the small Aurora model (smoke tests; base weights only)",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    inits = [int(i) for i in args.inits.split(",")]
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    data = LazyCMIP6Data(args.data_path)
    last_needed = max(inits) + args.history_t + args.steps
    if last_needed > len(data) - 1:
        raise ValueError(
            f"Rollout needs timestep {last_needed} but data has only {len(data)} steps; "
            "reduce --steps or choose earlier --inits"
        )
    save_fields_init = (
        args.save_fields_init if args.save_fields_init is not None else inits[0]
    )

    nn_models: dict[str, torch.nn.Module] = {}
    for name in model_names:
        if name in ("base", "finetuned"):
            LOGGER.info("Loading %s model...", name)
            nn_models[name] = build_model(name, args.checkpoint, args.small).to(device)
    clim = climatology_batch(data) if "climatology" in model_names else None

    records: list[dict] = []
    truth_fields: dict = {}
    for init in inits:
        init_start = time.time()
        input_batch = data.batch_at(init, t=args.history_t)
        init_time = input_batch.metadata.time[0]
        persistence = data.truth_at(init + args.history_t)
        rollouts = {
            name: rollout(model, input_batch.to(device), steps=args.steps)
            for name, model in nn_models.items()
        }
        field_records: dict[str, dict] = {name: {} for name in model_names}
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                lead_hours = 6 * step
                truth = data.truth_at(init + args.history_t + step)
                for name in model_names:
                    if name in rollouts:
                        pred = next(rollouts[name]).to("cpu")
                    elif name == "persistence":
                        pred = persistence
                    else:
                        pred = clim
                    for rec in compute_metrics(pred, truth):
                        records.append(
                            {
                                "model": name,
                                "init_time": init_time,
                                "lead_hours": lead_hours,
                                **rec,
                            }
                        )
                    if init == save_fields_init:
                        save_field_records(field_records[name], lead_hours, pred)
                if init == save_fields_init:
                    save_field_records(truth_fields, lead_hours, truth)
                LOGGER.info("init=%s lead=%sh done", init_time, lead_hours)
        if init == save_fields_init:
            for name in model_names:
                write_fields(args.out, name, field_records[name], data, init_time)
            write_fields(args.out, "truth", truth_fields, data, init_time)
        LOGGER.info("init=%s finished in %.1fs", init_time, time.time() - init_start)

        metrics = pd.DataFrame(records)
        metrics.to_csv(args.out / "metrics.csv", index=False)
    LOGGER.info("Wrote %s (%s rows)", args.out / "metrics.csv", len(records))


if __name__ == "__main__":
    main()
