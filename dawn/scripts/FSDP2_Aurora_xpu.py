import argparse
import os

import torch
from aurora import Aurora
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from fsdp_utils import inspect_mixed_precision, inspect_model
import torch.distributed as dist

from load_data import load_data
from load_batches import get_gt_batch, get_input_batch
from aurora_loss import mae


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
    os.environ["MASTER_PORT"] = "29500"  # your master port

    # initialize the process group by Intel® oneCCL Bindings for Pytorch\*
    dist.init_process_group("ccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    
    device = torch.device(f"xpu:{rank}")
    print(f"Setting device: {device}")
    torch.xpu.set_device(device)

    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)

    with torch.device("meta"):
        model = Aurora(use_lora=False, autocast=True)
        model.use_activation_checkpointing()

    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)
    
    if args.mixed_precision:
        inspect_mixed_precision(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Load data
    static_vars_ds, surf_vars_ds, atmos_vars_ds = load_data()

    for i in range(10):
        # Get input
        train_input = get_input_batch(i*3, static_vars_ds, surf_vars_ds, atmos_vars_ds)

        # Get output
        train_gt = get_gt_batch(i*3, static_vars_ds, surf_vars_ds, atmos_vars_ds)

        if args.explicit_prefetching:
            model.unshard()

        pred = model(train_input)
        loss = mae(pred, train_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    args = parser.parse_args()

    os.environ.pop("ZE_AFFINITY_MASK", None)
    WORLD_SIZE = torch.xpu.device_count()
    main(args)