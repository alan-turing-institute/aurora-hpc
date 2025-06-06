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

# PMI_RANK set by mpirun
RANK = os.environ["PMI_RANK"]
os.environ["RANK"] = RANK

# PMI_SIZE set by mpirun
WORLD_SIZE = os.environ["PMI_SIZE"]
assert WORLD_SIZE == "2"
os.environ["WORLD_SIZE"] = WORLD_SIZE

os.environ["MASTER_ADDR"] = "0.0.0.0"  # your master address
os.environ["MASTER_PORT"] = "29500"  # your master port


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
    if int(WORLD_SIZE) > 0:
        os.environ["RANK"] = str(RANK)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    else:
        # set the default rank and world size to 0 and 1
        os.environ["RANK"] = str(os.environ.get("RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))

    # Initialize the process group with ccl backend
    dist.init_process_group(backend="ccl")

    # For single-node distributed training, local_rank is the same as global rank
    local_rank = dist.get_rank()
    print(f"{local_rank=}, {type(local_rank)=}")
    print(f"{RANK=}")
    print(f"Available devices: {torch.xpu.device_count()}")

    # Only set device for distributed training on GPU
    device = torch.device(f"xpu:{local_rank}")
    # device = "xpu"
    print(f"Sending to device: {device}")
    # torch.xpu.set_device(device) # still errors
    model = Model().to(device)
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
