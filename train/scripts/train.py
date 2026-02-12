"""Fine tune Aurora weather model."""

print("importing...")
import argparse
import os
import re
import time
import warnings
from datetime import datetime as dt
from pathlib import Path
from typing import Iterator, Tuple

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import torch
import torch.nn as nn
from torch.distributed import all_gather, destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from aurora import Aurora
from aurora.model.swin3d import (
    Basic3DDecoderLayer,
    Basic3DEncoderLayer,
    Swin3DTransformerBackbone,
    Swin3DTransformerBlock,
)
from aurora_hpc.aurora_loss import mae
from aurora_hpc.dataset import AuroraDataset, aurora_collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--xpu", action="store_true", help="boolean of whether to use xpu")
parser.add_argument(
    "--shard", action="store_true", help="whether to use full_shard strategy"
)
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
)
parser.add_argument(
    "--tb_logdir",
    type=str,
    default="runs/aurora_train",
    help="TensorBoard log directory",
)
parser.add_argument(
    "--tb_log_interval",
    type=int,
    default=1,
    help="Log scalar/system metrics every N batches",
)
parser.add_argument(
    "--tb_hist_interval",
    type=int,
    default=50,
    help="Log gradient histograms every N batches (0 disables)",
)
args = parser.parse_args()

if args.xpu:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch  # has side-effects

    # unset affinity mask
    os.environ.pop("ZE_AFFINITY_MASK", None)

    # PMI_SIZE set by mpirun
    WORLD_SIZE = int(os.environ["PMI_SIZE"])
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)

    # PMI_RANK set by mpirun
    RANK = os.environ["PMI_RANK"]
    os.environ["RANK"] = RANK

    # MPI_LOCALRANKID provenance unknown
    LOCAL_RANK = int(os.environ["MPI_LOCALRANKID"])
    print(f"{LOCAL_RANK=}")

    # get the master address
    numbers = re.compile("\d+")
    nodelist_env = os.getenv("SLURM_JOB_NODELIST")

    # e.g. "pvc-s-[24-25]"
    try:
        # If we're running on >1 node, we should set the MASTER_ADDR
        # to the hostname of rank 0.
        prefix = nodelist_env[0 : nodelist_env.index("[")]
        nodelist = tuple(prefix + x for x in numbers.findall(nodelist_env))
        master_addr = nodelist[0]
    except ValueError:
        # We must be running on a single node.
        master_addr = "0.0.0.0"

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29876"
    USE_SUBDEVICES = os.environ.get("USE_SUBDEVICES", False)

else:
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])


def main(download_path: str, shard: bool, xpu: bool = False):
    if xpu:
        comms_backend = "ccl"
        device_type = "xpu"
    else:
        comms_backend = "nccl"
        device_type = "cuda"

    time_start_total = time.time()
    print(f"Script start time: {dt.now()}")
    is_main_process = int(RANK) == 0

    # print("Initialising process group with backend", comms_backend, flush=True)
    # device = f"{device_type}:{LOCAL_RANK}"
    # ToDo Run 2 or more processes.
    # init_process_group(
    #    world_size=int(WORLD_SIZE),
    #    rank=int(RANK),
    #    backend=comms_backend,
    # )

    # device = f"{device_type}:{LOCAL_RANK}"
    device = "cuda"
    print(f"Using {device=}")

    print(f"Start time loading model: {dt.now()}")
    print("loading model...")
    model = Aurora(
        use_lora=False,  # Model was not fine-tuned.
        autocast=True,  # Use AMP.
    )
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    model.to(device)
    # if not xpu:
    #    torch.cuda.set_device(LOCAL_RANK)
    print(f"End time loading model: {dt.now()}")

    download_path = Path(download_path)

    policy = ModuleWrapPolicy(
        {Swin3DTransformerBackbone, Basic3DEncoderLayer, Basic3DDecoderLayer}
    )

    print("preparing model...")
    model.configure_activation_checkpointing()
    #    model = FSDP(
    #        model,
    #        device_id=LOCAL_RANK,
    #        use_orig_params=True,
    #        sharding_strategy=(
    #            ShardingStrategy.FULL_SHARD if shard else ShardingStrategy.NO_SHARD
    #        ),
    #        auto_wrap_policy=policy if shard else None,
    #    )
    #    model.train()
    #
    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())
    writer = SummaryWriter(log_dir=args.tb_logdir) if is_main_process else None

    time_start_loading_data = time.time()
    print(f"Start time loading data: {dt.now()}")
    print("loading data...")
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_data=Path("static.nc"),
        # surface_data=Path("2023-01-surface-level-36.nc"),
        surface_data=Path("2023-01-surface-level-36.nc"),
        # atmos_data=Path("2023-01-atmospheric.nc"),
        atmos_data=Path("2023-01-atmospheric-36.nc"),
        len_max=32,
    )
    time_end_loading_data = time.time()
    print(f"End time loading data: {dt.now()}")
    print(f"Time loading data: {time_end_loading_data - time_start_loading_data}")

    # sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=False,  # We don't need to shuffle.
        # sampler=sampler,
        collate_fn=aurora_collate_fn,
    )

    times = []
    target_global_batch = 8
    accum_steps = max(1, (target_global_batch + WORLD_SIZE - 1) // WORLD_SIZE)
    print(f"Using gradient accumulation: {accum_steps=}, {target_global_batch=}")
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    time_start = time.time()
    for batch, (X, y) in enumerate(data_loader):
        step_start = time.time()
        print(f"batch {batch}...")

        y = y.to(device)
        X = X.to(device)
        print(f"finished X and y to device: {time.time()-time_start}")

        from contextlib import nullcontext

        # with torch.autocast(device_type=device_type):
        with nullcontext():
            print("performing forward pass...")
            pred = model(X)
            print(f"finished model forward: {time.time()-time_start}")

            # only one of these is necessary
            # pred = pred.to(device)
            # print(f"finished pred to device: {time.time()-time_start}")

            # mean absolute error of one variable
            print("calculating loss...")

            # Todo: Are pred's of type PyTree and does it matter?
            loss = mae(pred, y)
            loss_for_backward = loss / accum_steps
            print(f"finished loss calc: {time.time()-time_start}")
            # if torch.isnan(loss) or torch.isinf(loss):
            #    print("Loss is NaN or Inf!")
            #    print(f"pred has NaN: {torch.isnan(pred).any()}")
            #    print(f"y has NaN: {torch.isnan(y).any()}")

        print("performing backward pass...")
        loss_for_backward.backward()
        print(f"finished loss backward: {time.time()-time_start}")

        should_log = writer is not None and (
            args.tb_log_interval > 0 and batch % args.tb_log_interval == 0
        )
        if should_log:
            global_step = batch
            writer.add_scalar("train/loss", loss.detach().item(), global_step)
            writer.add_scalar(
                "train/loss_scaled_for_backward",
                loss_for_backward.detach().item(),
                global_step,
            )
            writer.add_scalar(
                "train/loss_is_finite",
                float(torch.isfinite(loss.detach()).item()),
                global_step,
            )
            writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], global_step)

            grad_l2_norm_sq = 0.0
            grad_max_abs = 0.0
            grad_nonfinite_elems = 0.0
            grad_total_elems = 0.0
            param_l2_norm_sq = 0.0
            for _, param in model.named_parameters():
                param_l2_norm_sq += float(
                    torch.sum(param.detach().float() * param.detach().float()).item()
                )
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                grad_l2_norm_sq += float(torch.sum(grad.float() * grad.float()).item())
                grad_max_abs = max(grad_max_abs, float(grad.abs().max().item()))
                grad_nonfinite_elems += float((~torch.isfinite(grad)).sum().item())
                grad_total_elems += float(grad.numel())
            grad_l2_norm = grad_l2_norm_sq**0.5
            grad_nonfinite_frac = (
                grad_nonfinite_elems / grad_total_elems if grad_total_elems > 0 else 0.0
            )
            writer.add_scalar("grad/l2_norm", grad_l2_norm, global_step)
            writer.add_scalar("grad/max_abs", grad_max_abs, global_step)
            writer.add_scalar(
                "grad/nonfinite_fraction", grad_nonfinite_frac, global_step
            )
            writer.add_scalar("model/param_l2_norm", param_l2_norm_sq**0.5, global_step)

            x_data = _batch_to_tensor(X).float()
            y_data = _batch_to_tensor(y).float()
            writer.add_scalar(
                "data/input_nan_fraction",
                float(torch.isnan(x_data).float().mean().item()),
                global_step,
            )
            writer.add_scalar(
                "data/input_inf_fraction",
                float(torch.isinf(x_data).float().mean().item()),
                global_step,
            )
            writer.add_scalar(
                "data/input_zero_fraction",
                float((x_data == 0).float().mean().item()),
                global_step,
            )
            writer.add_scalar(
                "data/target_nan_fraction",
                float(torch.isnan(y_data).float().mean().item()),
                global_step,
            )
            writer.add_scalar(
                "data/target_inf_fraction",
                float(torch.isinf(y_data).float().mean().item()),
                global_step,
            )
            writer.add_scalar(
                "data/target_zero_fraction",
                float((y_data == 0).float().mean().item()),
                global_step,
            )

            if torch.cuda.is_available():
                writer.add_scalar(
                    "system/cuda_memory_allocated_mb",
                    torch.cuda.memory_allocated() / (1024**2),
                    global_step,
                )
                writer.add_scalar(
                    "system/cuda_memory_reserved_mb",
                    torch.cuda.memory_reserved() / (1024**2),
                    global_step,
                )
                writer.add_scalar(
                    "system/cuda_max_memory_allocated_mb",
                    torch.cuda.max_memory_allocated() / (1024**2),
                    global_step,
                )

            if args.tb_hist_interval > 0 and batch % args.tb_hist_interval == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(
                            f"grad_hist/{name}", param.grad.detach(), global_step
                        )

        micro_step = batch + 1
        is_accum_boundary = micro_step % accum_steps == 0
        is_last_batch = micro_step == len(data_loader)
        if is_accum_boundary or is_last_batch:
            print("optimizing...")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            print(f"finished optimizer step: {time.time()-time_start}")
            if writer is not None:
                writer.add_scalar("optim/step", optimizer_steps, batch)

        time_end = time.time()
        print(f"Time for 1 iteration: {time_end - time_start}")
        if should_log:
            writer.add_scalar("timing/iter_seconds", time_end - step_start, batch)
        times.append(time_end - time_start)
        time_start = time.time()

    if writer is not None:
        writer.flush()
        writer.close()

    print("done")
    exit(0)
    times = torch.Tensor(times).to(device)
    gathered_times = [torch.zeros(times.shape).to(device) for _ in range(WORLD_SIZE)]
    all_gather(gathered_times, times)

    if int(RANK) == 0:
        avg_time = sum([sum(t[1:]) for t in gathered_times]) / sum(
            [len(times[1:]) for t in gathered_times]
        )
        print(f"Average time per epoch (ignoring first): {avg_time} seconds")
        print(f"Effective time for an epoch: {avg_time / WORLD_SIZE} seconds")
        print(f"Equivalent training speed: {WORLD_SIZE / avg_time} epochs per seconds")
        total_time = sum([sum(t) for t in gathered_times])
        total_no_epochs = sum([len(t) for t in gathered_times])
        print(f"Total time for {total_no_epochs} epochs: {total_time}")

        time_end_total = time.time()
        print(f"Total time: {time_end_total - time_start_total}")

    destroy_process_group()
    print("done")


def _iter_batch_tensors(batch_obj) -> Iterator[Tuple[str, torch.Tensor]]:
    if isinstance(batch_obj, torch.Tensor):
        yield "tensor", batch_obj
        return

    for group_name in ("surf_vars", "static_vars", "atmos_vars"):
        group = getattr(batch_obj, group_name, None)
        if isinstance(group, dict):
            for name, value in group.items():
                if isinstance(value, torch.Tensor):
                    yield f"{group_name}/{name}", value


def _batch_to_tensor(batch_obj) -> torch.Tensor:
    tensors = [t.reshape(-1) for _, t in _iter_batch_tensors(batch_obj)]
    if not tensors:
        return torch.zeros(1)
    return torch.cat(tensors)


main(args.download_path, shard=args.shard, xpu=args.xpu)
