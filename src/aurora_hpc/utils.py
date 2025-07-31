import os
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import which
from subprocess import run
from typing import Any

import torch


class UsageMetrics(ABC):
    @abstractmethod
    def get_metrics(self) -> list[dict[str, Any]]:
        """Get device metrics."""

    @classmethod
    @abstractmethod
    def are_available(cls) -> bool:
        """Whether metrics are available."""


class DeviceIdError(Exception):
    """Exception raised when a device ID is not found in the metrics file."""

    def __init__(self, device_id: int):
        super().__init__(f"Device ID {device_id} not found in the metrics file.")
        self.device_id = device_id


class XpuMetricsV1(UsageMetrics):
    """Note that we expect metrics to be streaming to a file already."""

    def __init__(self, *args, metrics_file: Path = Path("xpu_metrics.txt"), **kwargs):
        """Will raise if the file does not exist."""
        super().__init__(*args, **kwargs)
        self.metrics_file = metrics_file
        if self.metrics_file.exists():
            with self.metrics_file.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        self.headers = first_line.split(",")

    @classmethod
    def are_available(cls) -> bool:
        """Check whether XPU is available."""
        return which("xpu-smi") is not None

    @staticmethod
    def get_previous_line(f) -> str:
        """Get the previous line of the metrics file."""
        # Read backwards until we find a newline
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)

        # Read forward to get the last line
        line = f.readline()

        # Leave the cursor where it should be for the next call
        f.seek(-1 * (len(line) + 2), os.SEEK_CUR)

        return line.decode("utf-8").strip()

    def get_last_device_entry(self, device_id) -> list[str]:
        """Get the last line of the metrics file."""
        with self.metrics_file.open("rb") as f:
            # Move to the end of the file
            f.seek(-1, os.SEEK_END)

            try:
                while (line := self.get_previous_line(f).split(",")) and line[1] != str(
                    device_id
                ):
                    continue
            except OSError:
                raise DeviceIdError(device_id)

            return line

    def get_metrics(self) -> list[dict[str, Any]]:
        """Get XPU metrics."""
        results = []
        try:
            for device_id in range(4):
                results.append(
                    {
                        k: v
                        for k, v in zip(
                            self.headers, self.get_last_device_entry(device_id)
                        )
                    }
                )
        except DeviceIdError:
            pass
        return results


class XpuMetricsV2(UsageMetrics):
    """Note that we expect metrics to be streaming to a file already."""

    @classmethod
    def are_available(cls) -> bool:
        """Check whether XPU is available."""
        return which("xpu-smi") is not None

    def get_metrics(self) -> list[dict[str, str]]:
        """Get XPU metrics."""
        completed_process = run(
            ["xpu-smi", "dump", "--metrics", "0,1", "--device", "-1", "-n", "1"],
            capture_output=True,
            check=True,
            text=True,
        )
        output = completed_process.stdout.splitlines()
        headers = [x.strip() for x in output[0].split(",")]
        return [
            {k: v.strip() for k, v in zip(headers, line.split(","))}
            for line in output[1:]
        ]


class CudaMetrics(UsageMetrics):
    @classmethod
    def are_available(cls) -> bool:
        """Check whether CUDA is available."""
        return torch.cuda.is_available()

    def get_metrics(self) -> list[dict[str, Any]]:
        return [
            {
                "gpu_util": torch.cuda.utilization(),
            }
        ]
