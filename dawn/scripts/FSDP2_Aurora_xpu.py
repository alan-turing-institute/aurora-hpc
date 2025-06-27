import argparse
import os
from functools import partial

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
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard

from aurora import Aurora

# PMI_RANK set by mpirun
RANK = os.environ["PMI_RANK"]
os.environ["RANK"] = RANK

# PMI_SIZE set by mpirun
WORLD_SIZE = os.environ["PMI_SIZE"]
os.environ["WORLD_SIZE"] = WORLD_SIZE


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
    mesh = DeviceMesh("xpu", list(range(int(WORLD_SIZE))), mesh_dim_names=["model"])

    device = torch.device(f"xpu:{RANK}")
    print(f"Using device: {device}")

    print("loading model...")
    model = Aurora(
        use_lora=False,
        autocast=True,
    ).to(device)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    model.configure_activation_checkpointing()

    # Load data
    static_vars_ds, surf_vars_ds, atmos_vars_ds = load_data()

    # Get input
    print(f"batching with {(int(RANK) * 2) + 1=}")
    batch = get_input_batch(
        (int(RANK) * 2) + 1, static_vars_ds, surf_vars_ds, atmos_vars_ds
    )

    shard_from_local = partial(
        DTensor.from_local, device_mesh=mesh, placements=[Replicate()]
    )
    # batch_sharded = batch._fmap(shard_from_local)

    # Get output
    train_gt = get_gt_batch(
        (int(RANK) * 2) + 1, static_vars_ds, surf_vars_ds, atmos_vars_ds
    )
    # train_gt_sharded = train_gt._fmap(shard_from_local)

    fsdp_kwargs = {"mesh": mesh}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    print("Sharding...")
    for _, module in model.named_modules():
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    print("Inspecting model..")
    inspect_model(model)

    if args.mixed_precision:
        inspect_mixed_precision(model)

    model.train()

    # AdamW, as used in the paper.
    optim = torch.optim.AdamW(model.parameters())

    for _ in range(2):
        optim.zero_grad()

        # Forward pass
        print("Performing forward pass...")
        pred = model.forward(batch)

        loss = mae(pred, train_gt)
        print("Running loss backward")
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
