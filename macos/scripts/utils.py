from abc import ABC, abstractmethod
from pathlib import Path
from shutil import which
from typing import Any

import torch


class UsageMetrics(ABC):
    @classmethod
    @abstractmethod
    def get_metrics(cls) -> dict[str, Any]:
        """Get device metrics."""

    @abstractmethod
    def are_available(self) -> bool:
        """Whether metrics are available."""


class XpuMetrics(UsageMetrics):
    """Note that we expect metrics to be streaming to a file already."""

    def __init__(self, *args, metrics_file: Path = Path("xpu_metrics.txt"), **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_file = metrics_file
        self.first_line = None
        if self.metrics_file.exists():
            with self.metrics_file.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            # if not first_line:
            #     raise ValueError(f"Metrics file {self.metrics_file} is empty.")
            self.first_line = first_line

    @classmethod
    def are_available(cls) -> bool:
        """Check whether XPU is available."""
        return which("xpu-smi") is not None

    def get_metrics(self) -> dict[str, Any]:
        """Get XPU metrics."""
        # Get the first and last lines of xpu_metrics.txt
        with self.metrics_file.open("r", encoding="utf-8") as f:
            first_line = None
            for line in f:
                if first_line is None:
                    first_line = line.strip()
                else:
                    last_line = line.strip()
        return {first_line: last_line}


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
    def are_available(self) -> bool:
        """CPU metrics are always available."""
        return True

    def get_metrics(cls) -> dict[str, Any]:
        # todo
        return {
            "cpu_util": 50,  # Placeholder value
            "cpu_temp": 60,  # Placeholder value
        }
