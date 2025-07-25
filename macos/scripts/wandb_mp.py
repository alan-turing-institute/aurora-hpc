import multiprocessing as mp
import time

import wandb


def gpu_monitor(queue):
    while True:
        gpu_util = get_fake_gpu_util()  # Replace with nvidia-smi or pynvml
        ts = time.time()
        queue.put({"gpu_util": gpu_util, "timestamp": ts})
        time.sleep(5)


def main():
    wandb.init(project="your_project")
    queue = mp.Queue()

    proc = mp.Process(target=gpu_monitor, args=(queue,), daemon=True)
    proc.start()

    step = 0
    while True:
        while not queue.empty():
            data = queue.get()
            wandb.log(
                {"gpu_util": data["gpu_util"], "logged_at": data["timestamp"]},
                step=step,
            )
            step += 1
        time.sleep(1)


def get_fake_gpu_util():
    # Replace with actual GPU util query
    return int(100 * (0.5 + 0.5 * time.time() % 1))


if __name__ == "__main__":
    main()
