"""Fine tune Aurora weather model."""

import argparse
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from aurora import Aurora, AuroraHighRes, AuroraSmall
from aurora_hpc.aurora_loss import mae
from aurora_hpc.dataset import AuroraDataset, aurora_collate_fn


@dataclass
class TrainConfig:
    download_path: Path
    xpu: bool
    shard: bool
    tb_logdir: str
    tb_log_interval: int
    tb_hist_interval: int
    target_global_batch: int
    len_max: Optional[int]
    epochs: int
    learning_rate: float
    model_size: str
    checkpoint_repo: str
    checkpoint_name: Optional[str]
    timing_log_path: Optional[Path]


@dataclass
class RuntimeContext:
    world_size: int
    rank: int
    local_rank: int
    comms_backend: str
    device_type: str
    device: str
    is_main_process: bool
    master_addr: Optional[str] = None
    master_port: Optional[str] = None


def parse_train_config() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xpu", action="store_true", help="boolean of whether to use xpu"
    )
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
    parser.add_argument(
        "--target_global_batch",
        type=int,
        default=8,
        help="Target effective global batch size for gradient accumulation",
    )
    parser.add_argument(
        "--len_max",
        type=int,
        default=None,
        help="Maximum dataset length used by AuroraDataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to fine-tune for",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=1e-3,  # Match Torch default
        help="Learning rate used by AdamW optimizer",
    )
    parser.add_argument(
        "--model_size",
        choices=("base", "small", "highres"),
        default="base",
        help="Aurora model size to use",
    )
    parser.add_argument(
        "--checkpoint_repo",
        type=str,
        default="microsoft/aurora",
        help="HuggingFace repository for the model checkpoint",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Checkpoint filename; defaults depend on --model_size",
    )
    parser.add_argument(
        "--timing_log_path",
        type=Path,
        default=None,
        help="Optional file path for detailed per-batch timing logs",
    )
    args = parser.parse_args(sys.argv[1:])
    return TrainConfig(
        download_path=Path(args.download_path),
        xpu=args.xpu,
        shard=args.shard,
        tb_logdir=args.tb_logdir,
        tb_log_interval=args.tb_log_interval,
        tb_hist_interval=args.tb_hist_interval,
        target_global_batch=args.target_global_batch,
        len_max=args.len_max,
        epochs=args.epochs,
        learning_rate=args.lr,
        model_size=args.model_size,
        checkpoint_repo=args.checkpoint_repo,
        checkpoint_name=args.checkpoint_name,
        timing_log_path=args.timing_log_path,
    )


def _parse_master_addr(nodelist_env: Optional[str]) -> str:
    if not nodelist_env:
        return "0.0.0.0"
    numbers = re.compile(r"\d+")
    try:
        prefix = nodelist_env[0 : nodelist_env.index("[")]
        nodelist = tuple(prefix + x for x in numbers.findall(nodelist_env))
        return nodelist[0]
    except ValueError:
        return "0.0.0.0"


def resolve_runtime_context(
    cfg: TrainConfig, env: Optional[Dict[str, str]] = None
) -> RuntimeContext:
    env = dict(os.environ if env is None else env)

    if cfg.xpu:
        world_size = int(env["PMI_SIZE"])
        rank = int(env["PMI_RANK"])
        local_rank = int(env["MPI_LOCALRANKID"])
        master_addr = _parse_master_addr(env.get("SLURM_JOB_NODELIST"))
        master_port = "29876"
        comms_backend = "ccl"
        device_type = "xpu"
        device = f"{device_type}:{local_rank}"
    else:
        world_size = int(env.get("WORLD_SIZE", "1"))
        rank = int(env.get("RANK", "0"))
        local_rank = int(env.get("LOCAL_RANK", "0"))
        master_addr = None
        master_port = None
        comms_backend = "nccl"
        device_type = "cuda"
        device = "cuda"

    return RuntimeContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        comms_backend=comms_backend,
        device_type=device_type,
        device=device,
        is_main_process=rank == 0,
        master_addr=master_addr,
        master_port=master_port,
    )


def maybe_prepare_xpu_runtime(cfg: TrainConfig, ctx: RuntimeContext) -> None:
    if not cfg.xpu:
        return

    import intel_extension_for_pytorch  # noqa: F401
    import oneccl_bindings_for_pytorch  # noqa: F401

    os.environ.pop("ZE_AFFINITY_MASK", None)
    os.environ["WORLD_SIZE"] = str(ctx.world_size)
    os.environ["RANK"] = str(ctx.rank)
    if ctx.master_addr is not None:
        os.environ["MASTER_ADDR"] = ctx.master_addr
    if ctx.master_port is not None:
        os.environ["MASTER_PORT"] = ctx.master_port


def setup_loggers(
    cfg: TrainConfig, ctx: RuntimeContext
) -> Tuple[logging.Logger, Optional[logging.Logger]]:
    train_logger = logging.getLogger("aurora_hpc.train")
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    if not train_logger.handlers:
        train_handler = logging.StreamHandler()
        train_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        train_logger.addHandler(train_handler)

    timing_logger = None
    if cfg.timing_log_path is not None and ctx.is_main_process:
        timing_logger = logging.getLogger("aurora_hpc.timing")
        timing_logger.setLevel(logging.INFO)
        timing_logger.propagate = False
        timing_logger.handlers.clear()
        timing_handler = logging.FileHandler(cfg.timing_log_path)
        timing_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        timing_logger.addHandler(timing_handler)
        train_logger.info("Detailed timings will be written to %s", cfg.timing_log_path)
    return train_logger, timing_logger


def build_model(cfg: TrainConfig, ctx: RuntimeContext, logger: logging.Logger) -> Aurora:
    logger.info("Start time loading model: %s", dt.now())
    logger.info("Loading model")
    model_size_to_checkpoint = {
        "base": "aurora-0.25-pretrained.ckpt",
        "small": "aurora-0.25-small-pretrained.ckpt",
        "highres": "aurora-0.1-finetuned.ckpt",
    }
    checkpoint_name = cfg.checkpoint_name or model_size_to_checkpoint[cfg.model_size]

    if cfg.model_size == "base":
        model = Aurora(
            use_lora=False,
            autocast=True,
        )
    elif cfg.model_size == "small":
        model = AuroraSmall(
            autocast=True,
        )
    else:
        model = AuroraHighRes(
            autocast=True,
        )

    logger.info(
        "Using model_size=%s checkpoint=%s/%s",
        cfg.model_size,
        cfg.checkpoint_repo,
        checkpoint_name,
    )
    model.load_checkpoint(cfg.checkpoint_repo, checkpoint_name)
    model.to(ctx.device)
    model.configure_activation_checkpointing()
    logger.info("End time loading model: %s", dt.now())
    return model


def build_data_loader(cfg: TrainConfig, logger: logging.Logger) -> DataLoader:
    logger.info("Start time loading data: %s", dt.now())
    logger.info("Loading data")
    dataset = AuroraDataset(
        data_path=cfg.download_path,
        t=1,
        static_data=Path("static.nc"),
        surface_data=Path("2023-01-surface-level-36.nc"),
        atmos_data=Path("2023-01-atmospheric-36.nc"),
        len_max=cfg.len_max,
    )
    logger.info("End time loading data: %s", dt.now())
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=aurora_collate_fn,
    )


def build_optimizer(model: Aurora, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)


def setup_writer(cfg: TrainConfig, ctx: RuntimeContext) -> Optional[SummaryWriter]:
    if not ctx.is_main_process:
        return None
    return SummaryWriter(log_dir=cfg.tb_logdir)


def compute_accum_steps(target_global_batch: int, world_size: int) -> int:
    return max(1, (target_global_batch + world_size - 1) // world_size)


def should_step_optimizer(
    micro_step: int, accum_steps: int, total_micro_steps: int
) -> bool:
    return micro_step % accum_steps == 0 or micro_step == total_micro_steps


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


def _histogram_skip_reason(t: torch.Tensor) -> Optional[str]:
    if t.numel() == 0:
        return "empty tensor (numel=0)"
    if not bool(torch.isfinite(t).any().item()):
        return "all values are non-finite (NaN/Inf)"
    return None


def log_training_metrics(
    writer: SummaryWriter,
    model: Aurora,
    optimizer: torch.optim.Optimizer,
    X,
    y,
    loss: torch.Tensor,
    loss_for_backward: torch.Tensor,
    global_step: int,
    cfg: TrainConfig,
    histogram_skip_logged_once: bool,
    logger: logging.Logger,
) -> bool:
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
    writer.add_scalar("grad/nonfinite_fraction", grad_nonfinite_frac, global_step)
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

    if cfg.tb_hist_interval > 0 and global_step % cfg.tb_hist_interval == 0:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            skip_reason = _histogram_skip_reason(grad)
            if skip_reason is None:
                writer.add_histogram(f"grad_hist/{name}", grad, global_step)
                continue
            if not histogram_skip_logged_once:
                logger.warning(
                    "Skipping TensorBoard gradient histogram for '%s' at step %s: %s",
                    name,
                    global_step,
                    skip_reason,
                )
                histogram_skip_logged_once = True
    return histogram_skip_logged_once


def run_train_loop(
    cfg: TrainConfig,
    ctx: RuntimeContext,
    model: Aurora,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: Optional[SummaryWriter],
    logger: logging.Logger,
    timing_logger: Optional[logging.Logger],
) -> None:
    accum_steps = compute_accum_steps(cfg.target_global_batch, ctx.world_size)
    logger.info(
        "Using gradient accumulation: accum_steps=%s, target_global_batch=%s",
        accum_steps,
        cfg.target_global_batch,
    )
    optimizer_steps = 0
    histogram_skip_logged_once = False
    optimizer.zero_grad(set_to_none=True)

    for i in range(cfg.epochs):
        model.train()
        for batch, (X, y) in enumerate(data_loader):
            global_step = i * len(data_loader) + batch
            step_start = time.time()

            stage_start = time.time()
            y = y.to(ctx.device)
            X = X.to(ctx.device)
            to_device_seconds = time.time() - stage_start

            stage_start = time.time()
            pred = model(X)
            forward_seconds = time.time() - stage_start

            stage_start = time.time()
            loss = mae(pred, y)
            if not bool(torch.isfinite(loss).all().item()):
                loss_value = float(loss.detach().item())
                logger.error(
                    "Encountered non-finite loss at epoch=%s batch=%s step=%s: loss=%s",
                    i,
                    batch,
                    global_step,
                    loss_value,
                )
                raise RuntimeError(
                    "Stopping training because non-finite loss was detected "
                    f"(epoch={i}, batch={batch}, step={global_step}, loss={loss_value})."
                )
            loss_for_backward = loss / accum_steps
            loss_seconds = time.time() - stage_start

            stage_start = time.time()
            loss_for_backward.backward()
            backward_seconds = time.time() - stage_start

            should_log = writer is not None and (
                cfg.tb_log_interval > 0 and batch % cfg.tb_log_interval == 0
            )
            if should_log and writer is not None:
                histogram_skip_logged_once = log_training_metrics(
                    writer=writer,
                    model=model,
                    optimizer=optimizer,
                    X=X,
                    y=y,
                    loss=loss,
                    loss_for_backward=loss_for_backward,
                    global_step=global_step,
                    cfg=cfg,
                    histogram_skip_logged_once=histogram_skip_logged_once,
                    logger=logger,
                )

            micro_step = batch + 1
            optimizer_seconds = 0.0
            did_step_optimizer = False
            if should_step_optimizer(micro_step, accum_steps, len(data_loader)):
                stage_start = time.time()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                did_step_optimizer = True
                optimizer_seconds = time.time() - stage_start
                if writer is not None:
                    writer.add_scalar("optim/step", optimizer_steps, global_step)

            time_end = time.time()
            iter_seconds = time_end - step_start
            logger.info(
                "epoch=%s batch=%s step=%s loss=%.6f optimizer_step=%s",
                i,
                batch,
                global_step,
                float(loss.detach().item()),
                did_step_optimizer,
            )
            if timing_logger is not None:
                timing_logger.info(
                    "epoch=%s batch=%s step=%s to_device=%.3f forward=%.3f loss=%.3f backward=%.3f optimizer=%.3f total=%.3f",
                    i,
                    batch,
                    global_step,
                    to_device_seconds,
                    forward_seconds,
                    loss_seconds,
                    backward_seconds,
                    optimizer_seconds,
                    iter_seconds,
                )
            if should_log and writer is not None:
                writer.add_scalar("timing/iter_seconds", iter_seconds, global_step)


def finalize_writer(writer: Optional[SummaryWriter]) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()


def main(cfg: TrainConfig) -> None:
    ctx = resolve_runtime_context(cfg)
    logger, timing_logger = setup_loggers(cfg, ctx)
    logger.info("Script start time: %s", dt.now())
    maybe_prepare_xpu_runtime(cfg, ctx)
    logger.info("Using device=%s", ctx.device)

    model = build_model(cfg, ctx, logger)
    data_loader = build_data_loader(cfg, logger)
    optimizer = build_optimizer(model, cfg)
    writer = setup_writer(cfg, ctx)

    run_train_loop(
        cfg=cfg,
        ctx=ctx,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        writer=writer,
        logger=logger,
        timing_logger=timing_logger,
    )
    finalize_writer(writer)
    logger.info("Done")


if __name__ == "__main__":
    main(parse_train_config())
