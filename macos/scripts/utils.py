import os
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import which
from typing import Any

import torch


class UsageMetrics(ABC):
    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get device metrics."""

    @classmethod
    @abstractmethod
    def are_available(cls) -> bool:
        """Whether metrics are available."""


class XpuMetrics(UsageMetrics):
    """Note that we expect metrics to be streaming to a file already."""

    def __init__(self, *args, metrics_file: Path = Path("xpu_metrics.txt"), **kwargs):
        """Will raise if the file does not exist."""
        super().__init__(*args, **kwargs)
        self.metrics_file = metrics_file
        if self.metrics_file.exists():
            with self.metrics_file.open("r", encoding="utf-8") as f:
                self.first_line = f.readline().strip()

    @classmethod
    def are_available(cls) -> bool:
        """Check whether XPU is available."""
        return which("xpu-smi") is not None

    def get_last_line(self) -> str:
        """Get the last line of the metrics file."""
        with self.metrics_file.open("rb") as f:
            # Move to the end of the file
            f.seek(os.SEEK_END)

            # Read backwards until we find a newline
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)

            # Read forward to get the last line
            return f.readline().decode("utf-8").strip()

    def get_metrics(self) -> dict[str, Any]:
        """Get XPU metrics."""
        # Get the first and last lines of xpu_metrics.txt
        return {self.first_line: self.get_last_line()}


class CudaMetrics(UsageMetrics):
    @classmethod
    def are_available(cls) -> bool:
        """Check whether CUDA is available."""
        return torch.cuda.is_available()

    def get_metrics(self) -> dict[str, Any]:
        return {
            "gpu_util": torch.cuda.utilization(),
        }


class CpuMetrics(UsageMetrics):
    @classmethod
    def are_available(cls) -> bool:
        """CPU metrics are always available."""
        return True

    def get_metrics(self) -> dict[str, Any]:
        # todo
        return {
            "cpu_util": 50,  # Placeholder value
            "cpu_temp": 60,  # Placeholder value
        }
