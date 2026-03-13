from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import train


class TestTrainScriptRefactorTargets(unittest.TestCase):
    def test_compute_accum_steps_uses_ceiling_division(self):
        self.assertEqual(
            train.compute_accum_steps(target_global_batch=8, world_size=1), 8
        )
        self.assertEqual(
            train.compute_accum_steps(target_global_batch=8, world_size=4), 2
        )
        self.assertEqual(
            train.compute_accum_steps(target_global_batch=8, world_size=16), 1
        )
        self.assertEqual(
            train.compute_accum_steps(target_global_batch=9, world_size=4), 3
        )

    def test_should_step_optimizer_on_boundary_or_last_batch(self):
        self.assertTrue(
            train.should_step_optimizer(
                micro_step=2, accum_steps=2, total_micro_steps=5
            )
        )
        self.assertFalse(
            train.should_step_optimizer(
                micro_step=3, accum_steps=2, total_micro_steps=5
            )
        )
        self.assertTrue(
            train.should_step_optimizer(
                micro_step=5, accum_steps=2, total_micro_steps=5
            )
        )

    def test_batch_to_tensor_flattens_supported_batch_groups(self):
        batch = SimpleNamespace(
            surf_vars={"2t": torch.tensor([[1.0, 2.0]])},
            static_vars={"z": torch.tensor([3.0])},
            atmos_vars={"u": torch.tensor([[[4.0, 5.0]]])},
        )

        flattened = train._batch_to_tensor(batch)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(flattened, expected))

    def test_parse_train_config_reads_cli_args(self):
        cfg = train.parse_train_config(
            [
                "--download_path",
                "/tmp/data",
                "--tb_logdir",
                "runs/test",
                "--tb_log_interval",
                "7",
                "--tb_hist_interval",
                "11",
                "--shard",
            ]
        )

        self.assertEqual(str(cfg.download_path), "/tmp/data")
        self.assertEqual(cfg.tb_logdir, "runs/test")
        self.assertEqual(cfg.tb_log_interval, 7)
        self.assertEqual(cfg.tb_hist_interval, 11)
        self.assertTrue(cfg.shard)

    def test_resolve_runtime_context_non_xpu_from_env(self):
        cfg = train.TrainConfig(download_path=Path("."), xpu=False)
        ctx = train.resolve_runtime_context(
            cfg,
            env={"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "1"},
        )

        self.assertEqual(ctx.world_size, 4)
        self.assertEqual(ctx.rank, 2)
        self.assertEqual(ctx.local_rank, 1)
        self.assertEqual(ctx.device_type, "cuda")
        self.assertEqual(ctx.comms_backend, "nccl")

    def test_resolve_runtime_context_xpu_parses_master_addr(self):
        cfg = train.TrainConfig(download_path=Path("."), xpu=True)
        ctx = train.resolve_runtime_context(
            cfg,
            env={
                "PMI_SIZE": "8",
                "PMI_RANK": "3",
                "MPI_LOCALRANKID": "1",
                "SLURM_JOB_NODELIST": "pvc-s-[24-25]",
            },
        )

        self.assertEqual(ctx.world_size, 8)
        self.assertEqual(ctx.rank, 3)
        self.assertEqual(ctx.local_rank, 1)
        self.assertEqual(ctx.device_type, "xpu")
        self.assertEqual(ctx.comms_backend, "ccl")
        self.assertEqual(ctx.master_addr, "pvc-s-24")


if __name__ == "__main__":
    unittest.main()
