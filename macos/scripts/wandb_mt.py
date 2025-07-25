"""Test out Weights and Biases.

Remember to set `WANDB_DIR=wandb_logs`
to avoid having a local dir called "wandb".
"""

import queue
import random
import threading
import time

import wandb
from pynvml import *

q = queue.Queue()


def gpu_monitor():
    nvmlInit()
    while True:
        ts = time.time()
        # gpu_util = random.randint(0, 100)  # Simulate GPU utilization

        # nvmlSystemGetCudaDriverVersion()
        # nvmlDeviceGetCount()

        handle = nvmlDeviceGetHandleByIndex(0)
        gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu
        q.put({"gpu_util": gpu_util, "timestamp": ts})
        time.sleep(1)


def main_loop():
    wandb.init(project="your_project")
    step = 0
    for _ in range(10):
        while not q.empty():
            data = q.get()
            wandb.log(
                {"gpu_util": data["gpu_util"], "logged_at": data["timestamp"]},
                step=step,
            )
            step += 1
        time.sleep(2)


threading.Thread(target=gpu_monitor, daemon=True).start()
main_loop()
