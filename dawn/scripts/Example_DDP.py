"""
Use MPI as the launcher to start DDP on single node with multiple devices.

https://intel.github.io/intel-extension-for-pytorch/xpu/2.7.10+xpu/tutorials/features/DDP.html
"""

import os
import time

import intel_extension_for_pytorch as ipex  # has side effects
import oneccl_bindings_for_pytorch  # has side effects
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)


if __name__ == "__main__":
    # unset ZE_AFFINITY_MASK (this is being set somewhere?)
    os.environ.pop("ZE_AFFINITY_MASK", None)

    torch.xpu.manual_seed(123)  # set a seed number
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    if mpi_world_size > 0:
        os.environ["RANK"] = str(mpi_rank)
        os.environ["WORLD_SIZE"] = str(mpi_world_size)
    else:
        # set the default rank and world size to 0 and 1
        os.environ["RANK"] = str(os.environ.get("RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
    os.environ["MASTER_PORT"] = "29500"  # your master port

    # Initialize the process group with ccl backend
    dist.init_process_group(backend="ccl")

    # For single-node distributed training, local_rank is the same as global rank
    local_rank = dist.get_rank()
    print(f"local rank: {local_rank}")
    print(f"Available devices: {torch.xpu.device_count()}")

    # Only set device for distributed training on GPU
    device = torch.device(f"xpu:{local_rank}")
    # device = "xpu"
    print(f"Sending to device: {device}")
    torch.xpu.set_device(device)
    model = Model()
    if dist.get_world_size() > 1:
        model = DDP(model).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    times = []
    for i in range(3):
        start_time = time.time()
        optimizer.zero_grad()
        print("Runing Iteration: {} on device {}".format(i, device))
        input = torch.randn(2, 4).to(device)
        labels = torch.randn(2, 5).to(device)
        # forward
        print(
            f"[Rank {local_rank}] input shape: {input.shape}, input device: {input.device}"
        )
        print(
            f"[Rank {local_rank}] label shape: {labels.shape}, label device: {labels.device}"
        )
        print(f"[Rank {local_rank}] model device: {next(model.parameters()).device}")

        print("Runing forward: {} on device {}".format(i, device))
        res = model(input)
        # loss
        print("Runing loss: {} on device {}".format(i, device))
        L = loss_fn(res, labels)
        # backward
        print("Runing backward: {} on device {}".format(i, device))
        L.backward()
        # update
        print("Runing optim: {} on device {}".format(i, device))
        optimizer.step()
        end_time = time.time()
        print(f"Time for one epoch: {end_time - start_time}")
        times.append(end_time - start_time)

    print(f"Total time: {sum(times)}")
    print(f"Average time: {sum(times)/len(times)}")
