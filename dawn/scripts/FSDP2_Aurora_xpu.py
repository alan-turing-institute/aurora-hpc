import argparse
import os

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # has side-effects
import torch
import torch.distributed as dist
from aurora_loss import mae
from fsdp_utils import inspect_mixed_precision, inspect_model
from load_batches import get_gt_batch, get_input_batch
from load_data import load_data
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from aurora import Aurora

# PMI_RANK set by mpirun
RANK = os.environ["PMI_RANK"]
os.environ["RANK"] = RANK

# PMI_SIZE set by mpirun
WORLD_SIZE = os.environ["PMI_SIZE"]
assert WORLD_SIZE == "2"
os.environ["WORLD_SIZE"] = WORLD_SIZE
USE_SUBDEVICES = os.environ.get("USE_SUBDEVICES", False)


def setup():
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
    os.environ["MASTER_PORT"] = "29500"  # your master port

    # initialize the process group by Intel® oneCCL Bindings for Pytorch\*
    init_process_group(
        backend="ccl",
    )


def cleanup():
    dist.destroy_process_group()


def main(args):
    setup()

    device = torch.device(f"xpu:{RANK}")
    print(f"Using device: {device}")

    print("loading model...")
    model = Aurora(
        use_lora=False, 
        autocast=True,
    )
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    # Load data
    static_vars_ds, surf_vars_ds, atmos_vars_ds = load_data()

    # Get input
    print(f"batching with {(int(RANK) * 2) + 1=}")
    train_input = get_input_batch(
        (int(RANK) * 2) + 1, static_vars_ds, surf_vars_ds, atmos_vars_ds
    ).to(device)

    # Get output
    train_gt = get_gt_batch(
        (int(RANK) * 2) + 1, static_vars_ds, surf_vars_ds, atmos_vars_ds
    ).to(device)
    model.configure_activation_checkpointing()

    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    fully_shard(model, **fsdp_kwargs).to(device)
    inspect_model(model)

    if args.mixed_precision:
        inspect_mixed_precision(model)

    model.train()
    
    # AdamW, as used in the paper.
    optim = torch.optim.AdamW(model.parameters())

    for _ in range(2):
        # Forward pass
        print("Performing forward pass...")
        pred = model.forward(train_input)
        loss = mae(pred, train_gt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
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
    main(args)
