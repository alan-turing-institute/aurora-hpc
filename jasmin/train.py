"""Fine tune Aurora weather model."""

import argparse
import logging
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


def main(
    download_path: str | Path,
    epochs: int,
    log_dir: str | Path,
    small: bool,
    checkpoint_dir: str | Path,
    resume: bool,
    use_deepspeed: bool = False,
    ds_config: str | Path | None = None,
    use_adam8bit: bool = False,
    use_lora: bool = False,
    use_gpu: bool = False,
    use_chunk_and_check: bool = False,
):

    time_start_total = time.time()
    LOGGER.info("Script start time: %s", dt.now())

    assert Path(
        checkpoint_dir
    ).is_dir(), "Checkpoint dir does not exist or isn't a directory"

    if use_gpu and use_deepspeed:
        raise ValueError(
            "--gpu is only for non-DeepSpeed training; launch with DeepSpeed instead."
        )

    LOGGER.info("Start time loading model: %s", dt.now())
    if small:
        LOGGER.info("loading small model...")
        model = AuroraSmallPretrained(
            use_lora=use_lora, use_chunked_checkpointing=use_chunk_and_check
        )
    else:
        LOGGER.info("loading normal model...")
        model = AuroraPretrained(
            use_lora=use_lora, use_chunked_checkpointing=use_chunk_and_check
        )
    model.load_checkpoint(strict=not use_lora)
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
    dataset = AuroraDataset(data_path=download_path, t=1, use_dask=True)
    time_end_loading_data = time.time()
    LOGGER.info("End time loading data: %s", dt.now())
    LOGGER.info(
        "Time loading data: %ss", time_end_loading_data - time_start_loading_data
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=True,
        collate_fn=aurora_collate_fn,
        num_workers=0,
        worker_init_fn=partial(
            dataloader_worker_init,
            log_level=logging.getLevelName(LOGGER.getEffectiveLevel()),
        ),
    )
    num_batches = len(data_loader)
    remaining_epochs = max(epochs - start_epoch, 0)
    total_steps = remaining_epochs * num_batches
    LOGGER.info("Planned epochs: %s", epochs)
    LOGGER.info(
        "Starting epoch: %s", start_epoch + 1 if remaining_epochs else epochs + 1
    )
    LOGGER.info("Remaining epochs: %s", remaining_epochs)
    LOGGER.info("Planned batches per epoch: %s", num_batches)
    LOGGER.info("Planned remaining optimizer steps: %s", total_steps)
    LOGGER.info("Checkpoint directory: %s", checkpoint_dir)

    times = []

    with SummaryWriter(log_dir=log_dir) as writer:
        log_memory(writer, global_step, message_prefix="initial memory")
        for epoch in range(start_epoch, epochs):
            LOGGER.info("epoch %s/%s...", epoch + 1, epochs)
            epoch_losses = []
            epoch_start = time.time()
            log_memory(
                writer,
                global_step,
                message_prefix=f"epoch {epoch + 1} start memory",
            )

            time_start = time.time()
            for batch, (X, y) in enumerate(data_loader, start=1):
                LOGGER.info(
                    "epoch=%s/%s batch=%s/%s", epoch + 1, epochs, batch, num_batches
                )
                y = y.to(device)

                if model_engine:
                    model_engine.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)

                LOGGER.debug("performing forward pass...")
                pred = model_engine(X) if model_engine else model(X)
                LOGGER.debug("finished model forward: %ss", time.time() - time_start)

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
                    num_batches,
                    global_step,
                    loss_value,
                )
                LOGGER.debug("finished loss calc: %ss", time.time() - time_start)

                LOGGER.debug("performing backward pass...")
                if model_engine:
                    model_engine.backward(loss)
                    model_engine.step()
                else:
                    loss.backward()
                    optimizer.step()
                LOGGER.debug("finished loss backward: %ss", time.time() - time_start)

                time_end = time.time()
                LOGGER.info("Time for 1 iteration: %ss", time_end - time_start)
                times.append(time_end - time_start)
                time_start = time.time()
                global_step += 1
                log_memory(
                    writer,
                    global_step,
                    message_prefix=f"after epoch {epoch + 1} batch {batch} memory",
                )

            if epoch_losses:
                epoch_loss = sum(epoch_losses) / len(epoch_losses)
                writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
                LOGGER.info("Average loss for epoch %s: %s", epoch + 1, epoch_loss)
            writer.add_scalar("train/epoch_seconds", time.time() - epoch_start, epoch)
            log_memory(
                writer,
                global_step,
                prefix="memory_epoch",
                message_prefix=f"epoch {epoch + 1} end memory",
            )
            if model_engine:
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
    LOGGER.info("Total time: %ss", time_end_total - time_start_total)

    LOGGER.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", "-d", help="path to download directory", required=True, type=Path
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
        help="Use chunk & checkpoint to reduce peak memory consumption.",
        action="store_false",
    )
    parser.add_argument(
        "--ds_config",
        help="Path to DeepSpeed config JSON",
        default=Path("ds_config.json"),
        type=Path,
    )
    # Required by the DeepSpeed launcher; ignored in single-process mode.
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_logging(args.log_level)
    main(
        args.data_path,
        args.epochs,
        args.log_dir,
        args.use_small,
        args.checkpoint_dir,
        args.resume,
        use_deepspeed=args.deepspeed,
        ds_config=args.ds_config,
        use_adam8bit=args.adam8bit,
        use_lora=args.lora,
        use_gpu=args.gpu,
        use_chunk_and_check=(not args.no_chunk),
    )
