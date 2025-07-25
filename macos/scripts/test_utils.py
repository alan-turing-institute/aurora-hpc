import os
import pathlib
import unittest
from unittest.mock import patch

import utils


class TestCudaMetrics(unittest.TestCase):

    def test_metrics_are_available(self):
        """CUDA metrics availability depends on the system."""
        with patch("torch.cuda.is_available", autospec=True, return_value=True):
            self.assertTrue(utils.CudaMetrics.are_available())

    def test_get_metrics(self):
        """CUDA metrics should return a dictionary with expected keys."""
        cuda_metrics = utils.CudaMetrics()
        with patch("torch.cuda.utilization", autospec=True, return_value=75):
            metrics = cuda_metrics.get_metrics()[0]
        self.assertEqual(75, metrics["gpu_util"])


class TestXpuMetricsV1(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def which(cmd):
            return "/path/to/xpu-smi" if cmd == "xpu-smi" else None

        self.which = which
        self.test_file = pathlib.Path("dummy_xpu_metrics.txt")

    def setUp(self):
        """Set up the test environment."""
        # Note that xpu-smi files don't end in a newline.
        text = "gpu,tile,util\n0,0,5\n0,1,3\n0,0,50\n0,1,30"
        self.test_file.write_text(text, encoding="utf-8")

    def tearDown(self):
        """Clean up the test environment."""
        self.test_file.unlink(missing_ok=True)

    def test_metrics_are_available(self):
        """XPU metrics availability depends on the system."""
        with patch(
            "utils.which",
            autospec=True,
        ):
            self.assertTrue(utils.XpuMetricsV1.are_available())

    def test_get_metrics(self):

        # 'dump', '--metrics', '0,1', '--device', '-1', '-n', '1'
        metrics = utils.XpuMetricsV1(metrics_file=self.test_file).get_metrics()
        self.assertEqual({"gpu": "0", "tile": "0", "util": "50"}, metrics[0])
        self.assertEqual({"gpu": "0", "tile": "1", "util": "30"}, metrics[1])

    def test_get_metrics_invalid_device(self):

        with self.assertRaises(utils.DeviceIdError) as context:
            utils.XpuMetricsV1(metrics_file=self.test_file).get_last_device_entry(99)
        self.assertEqual(
            "Device ID 99 not found in the metrics file.", str(context.exception)
        )

    def test_previous_line(self):
        xpu_metrics = utils.XpuMetricsV1(metrics_file=self.test_file)
        with self.test_file.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            line = xpu_metrics.get_previous_line(f)
            self.assertEqual("0,1,30", line)
            line = xpu_metrics.get_previous_line(f)
            self.assertEqual("0,0,50", line)
            line = xpu_metrics.get_previous_line(f)
            self.assertEqual("0,1,3", line)
            line = xpu_metrics.get_previous_line(f)
            self.assertEqual("0,0,5", line)


class TestXpuMetricsV2(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_path = os.environ["PATH"].split(":")

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        cls.old_path = os.environ["PATH"].split(":")
        os.environ["PATH"] = ":".join(["xpu_smi_dir/"] + cls.old_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        os.environ["PATH"] = ":".join(cls.old_path)

    def test_metrics_are_available(self):
        """XPU metrics availability depends on the system."""
        with patch(
            "utils.which",
            autospec=True,
        ):
            self.assertTrue(utils.XpuMetricsV2.are_available())

    def test_get_metrics(self):
        metrics = utils.XpuMetricsV2().get_metrics()
        reduced_metrics = [
            {
                k: v
                for k, v in metric.items()
                if k in ["Timestamp", "DeviceId", "TileId", "Compute Engine 0 (%)"]
            }
            for metric in metrics
        ]

        self.assertListEqual(
            [
                {
                    "Timestamp": "16:53:07.186",
                    "DeviceId": "0",
                    "TileId": "0",
                    "Compute Engine 0 (%)": "0.00",
                },
                {
                    "Timestamp": "16:53:07.187",
                    "DeviceId": "0",
                    "TileId": "1",
                    "Compute Engine 0 (%)": "0.00",
                },
            ],
            reduced_metrics,
        )

    def test_get_metrics_two(self):
        with patch("utils.run", autospec=True) as mock_run:
            utils.XpuMetricsV2().get_metrics()
            mock_run.assert_called_once_with(
                ["xpu-smi", "dump", "--metrics", "0,1", "--device", "-1", "-n", "1"],
                capture_output=True,
                text=True,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
