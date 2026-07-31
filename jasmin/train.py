"""Fine tune Aurora weather model."""

import argparse
import json
import logging
import os
import socket
import subprocess
import time
import warnings
from datetime import datetime as dt
from functools import partial
from pathlib import Path

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import psutil
import torch
from aurora_loss import mae
from dataset import AuroraDataset, aurora_collate_fn
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from aurora import AuroraPretrained, AuroraSmallPretrained

try:
    import deepspeed

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    import bitsandbytes as bnb

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

BYTES_PER_MIB = 1024 * 1024
LOGGER = logging.getLogger(__name__)


def configure_logging(level_name: str, *, force: bool = False) -> None:
    level_name = level_name.upper()
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(processName)s pid=%(process)d %(name)s: %(message)s",
        force=force,
    )


def dataloader_worker_init(worker_id: int, log_level: str) -> None:
    configure_logging(log_level, force=True)
    worker_info = get_worker_info()
    dataset_name = type(worker_info.dataset).__name__ if worker_info else "unknown"
    LOGGER.info("DataLoader worker %s started for dataset=%s", worker_id, dataset_name)


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("epoch-*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch-{epoch + 1:04d}.pt"
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"]), int(checkpoint["global_step"])


def memory_usage_mib() -> dict[str, float]:
    process = psutil.Process()
    main_rss = process.memory_info().rss
    children_rss = 0
    children = []
    children_available = 1.0
    try:
        children = process.children(recursive=True)
    except (psutil.Error, PermissionError):
        children_available = 0.0
    for child in children:
        try:
            children_rss += child.memory_info().rss
        except (psutil.Error, PermissionError):
            continue
    return {
        "main": main_rss / BYTES_PER_MIB,
        "children": children_rss / BYTES_PER_MIB,
        "total": (main_rss + children_rss) / BYTES_PER_MIB,
        "children_count": float(len(children)),
        "children_available": children_available,
    }


def log_memory(
    writer: SummaryWriter,
    step: int,
    *,
    prefix: str = "memory",
    message_prefix: str = "memory",
) -> None:
    memory = memory_usage_mib()
    writer.add_scalar(f"{prefix}/main_rss_mib", memory["main"], step)
    writer.add_scalar(f"{prefix}/children_rss_mib", memory["children"], step)
    writer.add_scalar(f"{prefix}/total_rss_mib", memory["total"], step)
    writer.add_scalar(f"{prefix}/children_count", memory["children_count"], step)
    writer.add_scalar(
        f"{prefix}/children_available", memory["children_available"], step
    )
    LOGGER.info(
        f"{message_prefix}: main={memory['main']:.1f} MiB, "
        f"children={memory['children']:.1f} MiB across {int(memory['children_count'])} "
        f"processes, total={memory['total']:.1f} MiB"
    )
    return memory


def configure_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_memory_usage_mib(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "max_allocated": 0.0,
            "max_reserved": 0.0,
        }
    return {
        "allocated": torch.cuda.memory_allocated(device) / BYTES_PER_MIB,
        "reserved": torch.cuda.memory_reserved(device) / BYTES_PER_MIB,
        "max_allocated": torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB,
        "max_reserved": torch.cuda.max_memory_reserved(device) / BYTES_PER_MIB,
    }


def log_gpu_memory(
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    *,
    prefix: str = "gpu_memory",
    message_prefix: str = "gpu memory",
) -> None:
    if device.type != "cuda":
        return
    memory = gpu_memory_usage_mib(device)
    writer.add_scalar(f"{prefix}/allocated_mib", memory["allocated"], step)
    writer.add_scalar(f"{prefix}/reserved_mib", memory["reserved"], step)
    writer.add_scalar(f"{prefix}/max_allocated_mib", memory["max_allocated"], step)
    writer.add_scalar(f"{prefix}/max_reserved_mib", memory["max_reserved"], step)
    LOGGER.info(
        f"{message_prefix}: allocated={memory['allocated']:.1f} MiB, "
        f"reserved={memory['reserved']:.1f} MiB, "
        f"max_allocated={memory['max_allocated']:.1f} MiB, "
        f"max_reserved={memory['max_reserved']:.1f} MiB"
    )


def log_step_perf(
    writer: SummaryWriter,
    step: int,
    *,
    data_wait_seconds: float,
    forward_seconds: float,
    backward_step_seconds: float,
    step_seconds: float,
    prefix: str = "perf",
) -> None:
    writer.add_scalar(f"{prefix}/data_wait_seconds", data_wait_seconds, step)
    writer.add_scalar(f"{prefix}/forward_seconds", forward_seconds, step)
    writer.add_scalar(f"{prefix}/backward_step_seconds", backward_step_seconds, step)
    writer.add_scalar(f"{prefix}/step_seconds", step_seconds, step)
    writer.add_scalar(f"{prefix}/steps_per_second", 1.0 / step_seconds, step)


def git_commit_info(repo_dir: Path) -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


def slurm_job_info() -> dict[str, str | None]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "qos": os.environ.get("SLURM_JOB_QOS"),
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "gpus": os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE"),
    }


def gpu_device_name(device: torch.device) -> str | None:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else None


def run_metadata(cli_args: dict, device: torch.device) -> dict:
    return {
        "hostname": socket.gethostname(),
        "gpu_name": gpu_device_name(device),
        "git": git_commit_info(Path(__file__).resolve().parent),
        "slurm": slurm_job_info(),
        "cli_args": cli_args,
    }


def summarize_run(
    *,
    device: str,
    epochs: int,
    max_steps: int | None,
    total_steps: int,
    model_load_seconds: float,
    data_load_seconds: float,
    iteration_seconds: list[float],
    total_seconds: float,
    peak_host_rss_mib: float,
    peak_gpu_allocated_mib: float | None,
    peak_gpu_reserved_mib: float | None,
    warmup_iterations: int = 1,
    metadata: dict | None = None,
) -> dict:
    timed_iterations = iteration_seconds[warmup_iterations:]
    if timed_iterations:
        avg_iteration_seconds = sum(timed_iterations) / len(timed_iterations)
        steps_per_second = 1.0 / avg_iteration_seconds
    else:
        avg_iteration_seconds = None
        steps_per_second = None
    return {
        "device": device,
        "epochs": epochs,
        "max_steps": max_steps,
        "total_steps": total_steps,
        "model_load_seconds": model_load_seconds,
        "data_load_seconds": data_load_seconds,
        "avg_iteration_seconds": avg_iteration_seconds,
        "steps_per_second": steps_per_second,
        "total_seconds": total_seconds,
        "peak_host_rss_mib": peak_host_rss_mib,
        "peak_gpu_allocated_mib": peak_gpu_allocated_mib,
        "peak_gpu_reserved_mib": peak_gpu_reserved_mib,
        "generated_at": dt.now().isoformat(timespec="seconds"),
        "metadata": metadata or {},
    }


def main(
    download_path: str | Path,
    epochs: int,
    log_dir: str | Path,
    small: bool,
    checkpoint_dir: str | Path,
    resume: bool,
    build_dir: str = "build",
    use_deepspeed: bool = False,
    ds_config: str | Path | None = None,
    use_adam8bit: bool = False,
    use_lora: bool = False,
    use_gpu: bool = False,
    use_chunk_and_check: bool = False,
    max_steps: int | None = None,
    skip_checkpoint: bool = False,
    use_tf32: bool = True,
    use_autocast: bool = True,
    hf_offline: bool = False,
    seed: int | None = None,
    cli_args: dict | None = None,
):

    time_start_total = time.time()
    LOGGER.info("Script start time: %s", dt.now())

    if seed is not None:
        torch.manual_seed(seed)
        LOGGER.info("Seeded torch RNG with seed=%s", seed)

    assert Path(
        checkpoint_dir
    ).is_dir(), "Checkpoint dir does not exist or isn't a directory"

    if use_gpu and use_deepspeed:
        raise ValueError(
            "--gpu is only for non-DeepSpeed training; launch with DeepSpeed instead."
        )

    configure_tf32(use_tf32)
    LOGGER.info("TensorFloat-32 (TF32): %s", "enabled" if use_tf32 else "disabled")

    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        LOGGER.info(
            "HF_HUB_OFFLINE=1: loading the checkpoint from the local Hugging Face "
            "Hub cache without a network check (fails if it isn't cached yet)."
        )

    time_start_loading_model = time.time()
    LOGGER.info("Start time loading model: %s", dt.now())
    if small:
        LOGGER.info("loading small model...")
        model = AuroraSmallPretrained(
            use_lora=use_lora,
            use_chunked_checkpointing=use_chunk_and_check,
            autocast=use_autocast,
        )
    else:
        LOGGER.info("loading normal model...")
        model = AuroraPretrained(
            use_lora=use_lora,
            use_chunked_checkpointing=use_chunk_and_check,
            autocast=use_autocast,
        )
    LOGGER.info(
        "BF16 autocast (backbone): %s", "enabled" if use_autocast else "disabled"
    )
    model.load_checkpoint(strict=not use_lora)
    time_end_loading_model = time.time()
    model_load_seconds = time_end_loading_model - time_start_loading_model
    LOGGER.info("End time loading model: %s", dt.now())

    if use_lora:
        for param in model.parameters():
            param.requires_grad_(False)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        LOGGER.info(
            "LoRA enabled: %s/%s trainable parameters (%.1f%%)",
            trainable,
            total,
            100 * trainable / total,
        )

    model.configure_activation_checkpointing()
    model.train()

    checkpoint_dir = Path(checkpoint_dir)
    start_epoch = 0
    global_step = 0
    device = torch.device("cpu")

    if use_deepspeed:
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("deepspeed is not installed; run: pip install deepspeed")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=str(ds_config),
        )
        device = model_engine.device
        if resume:
            _, client_state = model_engine.load_checkpoint(checkpoint_dir)
            if client_state:
                start_epoch = client_state.get("epoch", 0)
                global_step = client_state.get("global_step", 0)
                LOGGER.info(
                    "Resumed from deepspeed checkpoint: completed epochs=%s, global_step=%s",
                    start_epoch,
                    global_step,
                )
            else:
                LOGGER.info(
                    "No deepspeed checkpoint found in %s; starting from scratch.",
                    checkpoint_dir,
                )
    else:
        if use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("--gpu was requested, but CUDA is not available")
            device = torch.device("cuda")
        LOGGER.info("Using training device: %s", device)
        model.to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if use_adam8bit:
            if not BNB_AVAILABLE:
                raise RuntimeError(
                    "bitsandbytes is not installed; run: pip install bitsandbytes"
                )
            optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=3e-5, weight_decay=0.01
            )
            LOGGER.info("Using 8-bit AdamW optimizer")
        else:
            # AdamW, as used in the paper.
            optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)
        model_engine = None
        if resume:
            checkpoint_path = latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                LOGGER.info(
                    "No checkpoints found in %s; starting from scratch.", checkpoint_dir
                )
            else:
                start_epoch, global_step = load_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    device,
                )
                LOGGER.info(
                    "Resumed from %s: completed epochs=%s, global_step=%s",
                    checkpoint_path,
                    start_epoch,
                    global_step,
                )

    time_start_loading_data = time.time()
    LOGGER.info("Start time loading data: %s", dt.now())
    LOGGER.info("loading data...")
    dataset = AuroraDataset(
        data_path=download_path, t=1, use_dask=True, build_dir=build_dir
    )
    time_end_loading_data = time.time()
    data_load_seconds = time_end_loading_data - time_start_loading_data
    LOGGER.info("End time loading data: %s", dt.now())
    LOGGER.info("Time loading data: %ss", data_load_seconds)

    shuffle_generator = (
        torch.Generator().manual_seed(seed) if seed is not None else None
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=aurora_collate_fn,
        num_workers=0,
        worker_init_fn=partial(
            dataloader_worker_init,
            log_level=logging.getLevelName(LOGGER.getEffectiveLevel()),
        ),
    )
    num_batches = len(data_loader)
    batches_per_epoch = min(num_batches, max_steps) if max_steps else num_batches
    remaining_epochs = max(epochs - start_epoch, 0)
    total_steps = remaining_epochs * batches_per_epoch
    LOGGER.info("Planned epochs: %s", epochs)
    LOGGER.info(
        "Starting epoch: %s", start_epoch + 1 if remaining_epochs else epochs + 1
    )
    LOGGER.info("Remaining epochs: %s", remaining_epochs)
    LOGGER.info("Planned batches per epoch: %s", batches_per_epoch)
    LOGGER.info("Planned remaining optimizer steps: %s", total_steps)
    LOGGER.info("Checkpoint directory: %s", checkpoint_dir)

    times = []
    peak_host_rss_mib = 0.0

    with SummaryWriter(log_dir=log_dir) as writer:
        peak_host_rss_mib = max(
            peak_host_rss_mib,
            log_memory(writer, global_step, message_prefix="initial memory")["total"],
        )
        log_gpu_memory(writer, global_step, device, message_prefix="initial gpu memory")
        for epoch in range(start_epoch, epochs):
            LOGGER.info("epoch %s/%s...", epoch + 1, epochs)
            epoch_losses = []
            epoch_start = time.time()
            peak_host_rss_mib = max(
                peak_host_rss_mib,
                log_memory(
                    writer,
                    global_step,
                    message_prefix=f"epoch {epoch + 1} start memory",
                )["total"],
            )
            log_gpu_memory(
                writer,
                global_step,
                device,
                message_prefix=f"epoch {epoch + 1} start gpu memory",
            )

            time_iter_start = time.time()
            for batch, (X, y) in enumerate(data_loader, start=1):
                LOGGER.info(
                    "epoch=%s/%s batch=%s/%s",
                    epoch + 1,
                    epochs,
                    batch,
                    batches_per_epoch,
                )
                data_wait_seconds = time.time() - time_iter_start
                y = y.to(device)

                if model_engine:
                    model_engine.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)

                LOGGER.debug("performing forward pass...")
                time_forward_start = time.time()
                pred = model_engine(X) if model_engine else model(X)
                maybe_synchronize(device)
                forward_seconds = time.time() - time_forward_start
                LOGGER.debug("finished model forward: %ss", forward_seconds)

                # mean absolute error of one variable
                LOGGER.debug("calculating loss...")

                loss = mae(pred, y)
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                writer.add_scalar("train/loss", loss_value, global_step)
                LOGGER.info(
                    "epoch=%s/%s batch=%s/%s step=%s loss=%s",
                    epoch + 1,
                    epochs,
                    batch,
                    batches_per_epoch,
                    global_step,
                    loss_value,
                )

                LOGGER.debug("performing backward pass...")
                time_backward_start = time.time()
                if model_engine:
                    model_engine.backward(loss)
                    model_engine.step()
                else:
                    loss.backward()
                    optimizer.step()
                maybe_synchronize(device)
                backward_step_seconds = time.time() - time_backward_start
                LOGGER.debug("finished loss backward: %ss", backward_step_seconds)

                time_iter_end = time.time()
                step_seconds = time_iter_end - time_iter_start
                LOGGER.info("Time for 1 iteration: %ss", step_seconds)
                times.append(step_seconds)
                global_step += 1
                log_step_perf(
                    writer,
                    global_step,
                    data_wait_seconds=data_wait_seconds,
                    forward_seconds=forward_seconds,
                    backward_step_seconds=backward_step_seconds,
                    step_seconds=step_seconds,
                )
                peak_host_rss_mib = max(
                    peak_host_rss_mib,
                    log_memory(
                        writer,
                        global_step,
                        message_prefix=f"after epoch {epoch + 1} batch {batch} memory",
                    )["total"],
                )
                log_gpu_memory(
                    writer,
                    global_step,
                    device,
                    message_prefix=f"after epoch {epoch + 1} batch {batch} gpu memory",
                )
                time_iter_start = time.time()
                if max_steps is not None and batch >= max_steps:
                    LOGGER.info(
                        "Reached --max_steps=%s; stopping epoch %s early.",
                        max_steps,
                        epoch + 1,
                    )
                    break

            if epoch_losses:
                epoch_loss = sum(epoch_losses) / len(epoch_losses)
                writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
                LOGGER.info("Average loss for epoch %s: %s", epoch + 1, epoch_loss)
            writer.add_scalar("train/epoch_seconds", time.time() - epoch_start, epoch)
            peak_host_rss_mib = max(
                peak_host_rss_mib,
                log_memory(
                    writer,
                    global_step,
                    prefix="memory_epoch",
                    message_prefix=f"epoch {epoch + 1} end memory",
                )["total"],
            )
            log_gpu_memory(
                writer,
                global_step,
                device,
                prefix="gpu_memory_epoch",
                message_prefix=f"epoch {epoch + 1} end gpu memory",
            )
            if skip_checkpoint:
                LOGGER.info("Skipped checkpoint save (--skip_checkpoint)")
            elif model_engine:
                model_engine.save_checkpoint(
                    checkpoint_dir,
                    client_state={"epoch": epoch + 1, "global_step": global_step},
                )
                LOGGER.info("Saved deepspeed checkpoint: epoch=%s", epoch + 1)
            else:
                checkpoint_path = save_checkpoint(
                    checkpoint_dir,
                    epoch,
                    global_step,
                    model,
                    optimizer,
                )
                LOGGER.info("Saved checkpoint: %s", checkpoint_path)

    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
        LOGGER.info("Average time per iteration (ignoring first): %s seconds", avg_time)

    total_time = sum(times)
    LOGGER.info("Total time for %s epochs: %ss", epochs, total_time)

    time_end_total = time.time()
    total_seconds = time_end_total - time_start_total
    LOGGER.info("Total time: %ss", total_seconds)

    peak_gpu_allocated_mib = None
    peak_gpu_reserved_mib = None
    if device.type == "cuda":
        peak_gpu_allocated_mib = torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB
        peak_gpu_reserved_mib = torch.cuda.max_memory_reserved(device) / BYTES_PER_MIB

    summary = summarize_run(
        device=str(device),
        epochs=epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        model_load_seconds=model_load_seconds,
        data_load_seconds=data_load_seconds,
        iteration_seconds=times,
        total_seconds=total_seconds,
        peak_host_rss_mib=peak_host_rss_mib,
        peak_gpu_allocated_mib=peak_gpu_allocated_mib,
        peak_gpu_reserved_mib=peak_gpu_reserved_mib,
        metadata=run_metadata(cli_args or {}, device),
    )
    summary_path = Path(log_dir) / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Wrote benchmark summary: %s", summary_path)

    LOGGER.info("done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", "-d", help="path to download directory", required=True, type=Path
    )
    parser.add_argument(
        "--build_dir",
        default="build",
        help=(
            "path (relative to --data_path) to the regrid pipeline's run output "
            "— e.g. 'build/MPI-ESM1-2-LR_ssp585_r1i1p1f1_...'. regrid namespaces "
            "each run under build/<run_id>/, so this must be set explicitly to "
            "pick a specific source/experiment/member/window."
        ),
    )
    parser.add_argument(
        "--epochs", "-e", help="number of training epochs", default=1, type=int
    )
    parser.add_argument(
        "--log_dir",
        help="TensorBoard log directory",
        default=Path("runs") / dt.now().strftime("%Y%m%d-%H%M%S"),
        type=Path,
    )
    parser.add_argument(
        "--use_small",
        help="Use the small model (for debugging)",
        action="store_true",
    )
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory for epoch checkpoints",
        default=Path("checkpoints"),
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="Resume from the latest checkpoint in --checkpoint_dir",
        action="store_true",
    )
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Python logging level",
    )
    optimizer_group = parser.add_mutually_exclusive_group()
    optimizer_group.add_argument(
        "--deepspeed",
        help="Enable DeepSpeed ZeRO offloading",
        action="store_true",
    )
    optimizer_group.add_argument(
        "--adam8bit",
        help="Use 8-bit AdamW optimizer (bitsandbytes) instead of fp32 AdamW",
        action="store_true",
    )
    parser.add_argument(
        "--lora",
        help="Use LoRA for finetuning (freezes pretrained weights, trains only LoRA adapters)",
        action="store_true",
    )
    parser.add_argument(
        "--gpu",
        help="Use CUDA for non-DeepSpeed training. Defaults to CPU when omitted.",
        action="store_true",
    )
    parser.add_argument(
        "--no-chunk",
        help="Don't use chunk & checkpoint to reduce peak memory consumption.",
        action="store_true",
    )
    parser.add_argument(
        "--no-tf32",
        help="Disable TensorFloat-32 for matmul/cuDNN ops. Enabled by default "
        "(no effect on pre-Ampere GPUs or CPU); benchmarked ~3.3x faster with "
        "no accuracy validation done yet beyond sane, non-NaN losses.",
        action="store_true",
    )
    parser.add_argument(
        "--no-autocast",
        help="Disable BF16 torch.autocast on the model backbone (Aurora's own "
        "`autocast` option). Enabled by default; benchmarked ~3.2x faster and "
        "lower peak memory, and Aurora's docs call it important for "
        "fine-tuning, but we haven't validated convergence/quality yet.",
        action="store_true",
    )
    parser.add_argument(
        "--hf_offline",
        help="Load the checkpoint from the local Hugging Face Hub cache without "
        "the network freshness check (sets HF_HUB_OFFLINE=1). Only works once "
        "the checkpoint has been downloaded at least once.",
        action="store_true",
    )
    parser.add_argument(
        "--ds_config",
        help="Path to DeepSpeed config JSON",
        default=Path("ds_config.json"),
        type=Path,
    )
    parser.add_argument(
        "--max_steps",
        help="Cap the number of batches processed per epoch (for quick benchmark runs)",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--skip_checkpoint",
        help="Skip saving epoch checkpoints (for throwaway benchmark runs)",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        help="Seed torch's RNG and the DataLoader shuffle order, for reproducible "
        "comparisons across runs (e.g. matching data order between a baseline "
        "and a variant run). Unseeded (nondeterministic shuffle) by default.",
        default=None,
        type=int,
    )
    # Required by the DeepSpeed launcher; ignored in single-process mode.
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser


def cli_main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    cli_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    main(
        args.data_path,
        args.epochs,
        args.log_dir,
        args.use_small,
        args.checkpoint_dir,
        args.resume,
        build_dir=args.build_dir,
        use_deepspeed=args.deepspeed,
        ds_config=args.ds_config,
        use_adam8bit=args.adam8bit,
        use_lora=args.lora,
        use_gpu=args.gpu,
        use_chunk_and_check=(not args.no_chunk),
        use_tf32=(not args.no_tf32),
        use_autocast=(not args.no_autocast),
        hf_offline=args.hf_offline,
        max_steps=args.max_steps,
        skip_checkpoint=args.skip_checkpoint,
        seed=args.seed,
        cli_args=cli_args,
    )


if __name__ == "__main__":
    cli_main()
