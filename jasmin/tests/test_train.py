import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from train import (
    configure_tf32,
    git_commit_info,
    gpu_device_name,
    gpu_memory_usage_mib,
    log_gpu_memory,
    log_step_perf,
    maybe_synchronize,
    run_metadata,
    slurm_job_info,
    summarize_run,
)


def read_scalar(log_dir: Path, tag: str, step: int) -> float:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    (event,) = (e for e in accumulator.Scalars(tag) if e.step == step)
    return event.value


def scalar_tags(log_dir: Path) -> set[str]:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return set(accumulator.Tags()["scalars"])


class ConfigureTf32Tests(unittest.TestCase):
    def setUp(self):
        self.original_matmul = torch.backends.cuda.matmul.allow_tf32
        self.original_cudnn = torch.backends.cudnn.allow_tf32
        self.addCleanup(self._restore)

    def _restore(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original_matmul
        torch.backends.cudnn.allow_tf32 = self.original_cudnn

    def test_enables_tf32_on_both_backends(self):
        configure_tf32(True)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_disables_tf32_on_both_backends(self):
        configure_tf32(False)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)


class MaybeSynchronizeTests(unittest.TestCase):
    def test_no_op_on_cpu_device(self):
        self.assertIsNone(maybe_synchronize(torch.device("cpu")))


class GpuMemoryUsageMibTests(unittest.TestCase):
    def test_returns_zeroed_dict_when_not_cuda(self):
        memory = gpu_memory_usage_mib(torch.device("cpu"))
        self.assertEqual(
            memory,
            {
                "allocated": 0.0,
                "reserved": 0.0,
                "max_allocated": 0.0,
                "max_reserved": 0.0,
            },
        )


class LogGpuMemoryTests(unittest.TestCase):
    def test_writes_no_scalars_for_cpu_device(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with SummaryWriter(log_dir=tmpdir) as writer:
                log_gpu_memory(writer, 0, torch.device("cpu"))
            self.assertEqual(
                scalar_tags(Path(tmpdir))
                & {
                    "gpu_memory/allocated_mib",
                    "gpu_memory/reserved_mib",
                    "gpu_memory/max_allocated_mib",
                    "gpu_memory/max_reserved_mib",
                },
                set(),
            )


class LogStepPerfTests(unittest.TestCase):
    def test_writes_expected_scalar_tags_and_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with SummaryWriter(log_dir=tmpdir) as writer:
                log_step_perf(
                    writer,
                    3,
                    data_wait_seconds=0.1,
                    forward_seconds=0.2,
                    backward_step_seconds=0.3,
                    step_seconds=0.5,
                )
            log_dir = Path(tmpdir)
            self.assertAlmostEqual(
                read_scalar(log_dir, "perf/data_wait_seconds", 3), 0.1, places=5
            )
            self.assertAlmostEqual(
                read_scalar(log_dir, "perf/forward_seconds", 3), 0.2, places=5
            )
            self.assertAlmostEqual(
                read_scalar(log_dir, "perf/backward_step_seconds", 3), 0.3, places=5
            )
            self.assertAlmostEqual(
                read_scalar(log_dir, "perf/step_seconds", 3), 0.5, places=5
            )
            self.assertAlmostEqual(
                read_scalar(log_dir, "perf/steps_per_second", 3), 2.0, places=5
            )


class GpuDeviceNameTests(unittest.TestCase):
    def test_returns_none_when_not_cuda(self):
        self.assertIsNone(gpu_device_name(torch.device("cpu")))


class GitCommitInfoTests(unittest.TestCase):
    def test_reports_current_repo_commit_and_dirty_flag(self):
        info = git_commit_info(Path(__file__).resolve().parent)
        self.assertIsInstance(info["commit"], str)
        self.assertEqual(len(info["commit"]), 40)
        self.assertIsInstance(info["dirty"], bool)

    def test_returns_none_values_outside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info = git_commit_info(Path(tmpdir))
            self.assertIsNone(info["commit"])
            self.assertIsNone(info["dirty"])


class SlurmJobInfoTests(unittest.TestCase):
    def test_returns_none_values_outside_slurm(self):
        slurm_keys = [key for key in os.environ if key.startswith("SLURM_")]
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in slurm_keys:
                os.environ.pop(key, None)
            info = slurm_job_info()
        self.assertEqual(
            info,
            {
                "job_id": None,
                "partition": None,
                "account": None,
                "qos": None,
                "nodelist": None,
                "cpus_per_task": None,
                "mem_per_node": None,
                "gpus": None,
            },
        )

    def test_reads_slurm_environment_variables(self):
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "12345", "SLURM_JOB_PARTITION": "orchid"},
        ):
            info = slurm_job_info()
        self.assertEqual(info["job_id"], "12345")
        self.assertEqual(info["partition"], "orchid")


class RunMetadataTests(unittest.TestCase):
    def test_includes_hostname_gpu_name_git_slurm_and_cli_args(self):
        metadata = run_metadata({"epochs": 1}, torch.device("cpu"))
        self.assertIn("hostname", metadata)
        self.assertIsNone(metadata["gpu_name"])
        self.assertIn("commit", metadata["git"])
        self.assertIn("job_id", metadata["slurm"])
        self.assertEqual(metadata["cli_args"], {"epochs": 1})


class SummarizeRunTests(unittest.TestCase):
    def base_kwargs(self, **overrides):
        kwargs = dict(
            device="cpu",
            epochs=2,
            max_steps=5,
            total_steps=10,
            model_load_seconds=1.0,
            data_load_seconds=2.0,
            iteration_seconds=[10.0, 1.0, 1.0, 1.0],
            total_seconds=15.0,
            peak_host_rss_mib=123.0,
            peak_gpu_allocated_mib=None,
            peak_gpu_reserved_mib=None,
        )
        kwargs.update(overrides)
        return kwargs

    def test_computes_avg_excluding_warmup(self):
        summary = summarize_run(**self.base_kwargs())
        self.assertAlmostEqual(summary["avg_iteration_seconds"], 1.0)
        self.assertAlmostEqual(summary["steps_per_second"], 1.0)

    def test_handles_empty_iteration_seconds(self):
        summary = summarize_run(**self.base_kwargs(iteration_seconds=[]))
        self.assertIsNone(summary["avg_iteration_seconds"])
        self.assertIsNone(summary["steps_per_second"])

    def test_handles_all_iterations_as_warmup(self):
        summary = summarize_run(
            **self.base_kwargs(iteration_seconds=[10.0], warmup_iterations=1)
        )
        self.assertIsNone(summary["avg_iteration_seconds"])
        self.assertIsNone(summary["steps_per_second"])

    def test_passthrough_fields_and_generated_at(self):
        summary = summarize_run(
            **self.base_kwargs(
                peak_gpu_allocated_mib=456.0, peak_gpu_reserved_mib=789.0
            )
        )
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["epochs"], 2)
        self.assertEqual(summary["max_steps"], 5)
        self.assertEqual(summary["total_steps"], 10)
        self.assertEqual(summary["model_load_seconds"], 1.0)
        self.assertEqual(summary["data_load_seconds"], 2.0)
        self.assertEqual(summary["peak_host_rss_mib"], 123.0)
        self.assertEqual(summary["peak_gpu_allocated_mib"], 456.0)
        self.assertEqual(summary["peak_gpu_reserved_mib"], 789.0)
        self.assertIn("generated_at", summary)

    def test_defaults_metadata_to_empty_dict(self):
        summary = summarize_run(**self.base_kwargs())
        self.assertEqual(summary["metadata"], {})

    def test_passes_through_metadata(self):
        summary = summarize_run(**self.base_kwargs(metadata={"hostname": "gpuhost013"}))
        self.assertEqual(summary["metadata"], {"hostname": "gpuhost013"})


if __name__ == "__main__":
    unittest.main()
