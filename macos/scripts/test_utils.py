import pathlib
import unittest
from unittest.mock import patch

import utils


class TestCpuMetrics(unittest.TestCase):
    def test_metrics_are_available(self):
        """CPU metrics should always be available."""
        self.assertTrue(utils.CpuMetrics.are_available())

    def test_get_metrics(self):
        """CPU metrics should return a dictionary with expected keys."""
        cpu_metrics = utils.CpuMetrics()
        metrics = cpu_metrics.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("cpu_util", metrics)
        self.assertIn("cpu_temp", metrics)


class TestCudaMetrics(unittest.TestCase):
    def test_metrics_are_available(self):
        """CUDA metrics availability depends on the system."""
        with patch("torch.cuda.is_available", autospec=True, return_value=True):
            self.assertTrue(utils.CudaMetrics.are_available())

    def test_get_metrics(self):
        """CUDA metrics should return a dictionary with expected keys."""
        cuda_metrics = utils.CudaMetrics()
        with patch("torch.cuda.utilization", autospec=True, return_value=75):
            metrics = cuda_metrics.get_metrics()
        self.assertEqual(75, metrics["gpu_util"])


class TestXpuMetrics(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def which(cmd):
            return "/path/to/xpu-smi" if cmd == "xpu-smi" else None

        self.which = which
        self.test_file = pathlib.Path("dummy_xpu_metrics.txt")

    def setUp(self):
        """Set up the test environment."""
        self.test_file.write_text("x\n1", encoding="utf-8")

    def tearDown(self):
        """Clean up the test environment."""
        self.test_file.unlink(missing_ok=True)

    def test_metrics_are_available(self):
        """XPU metrics availability depends on the system."""
        with patch(
            "utils.which",
            autospec=True,
        ) as mock_which:
            mock_which = self.which
            self.assertTrue(utils.XpuMetrics.are_available())

    def test_get_metrics(self):

        # 'dump', '--metrics', '0,1', '--device', '-1', '-n', '1'
        metrics = utils.XpuMetrics(metrics_file=self.test_file).get_metrics()
        self.assertEqual({"x": "1"}, metrics)


if __name__ == "__main__":
    unittest.main()
