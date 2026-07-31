import unittest
from pathlib import Path
from unittest import mock

from hypothesis import given
from hypothesis import strategies as st

from train import cli_main

BASE_ARGV = ["--data_path", "/tmp/does-not-need-to-exist"]

# --deepspeed and --adam8bit share a mutually exclusive group, so they can
# never both be True in a single invocation.
OPTIMIZER_CHOICES = st.sampled_from([(False, False), (True, False), (False, True)])


class NoChunkFlagTests(unittest.TestCase):
    """Regression coverage for the --no-chunk store_false/store_true mixup.

    --no-chunk previously used action="store_false", which made chunking
    default to *disabled* unless --no-chunk was explicitly passed - the
    opposite of the intended behaviour.
    """

    @mock.patch("train.main")
    def test_chunking_enabled_by_default(self, mock_main):
        cli_main([*BASE_ARGV])
        self.assertTrue(mock_main.call_args.kwargs["use_chunk_and_check"])

    @mock.patch("train.main")
    def test_no_chunk_flag_disables_chunking(self, mock_main):
        cli_main([*BASE_ARGV, "--no-chunk"])
        self.assertFalse(mock_main.call_args.kwargs["use_chunk_and_check"])


class CliArgumentPropagationTests(unittest.TestCase):
    """Checks that parsed CLI flags reach main() with the expected values,
    across many combinations of the boolean flags."""

    @given(
        use_small=st.booleans(),
        resume=st.booleans(),
        optimizer_flags=OPTIMIZER_CHOICES,
        lora=st.booleans(),
        gpu=st.booleans(),
        no_chunk=st.booleans(),
        skip_checkpoint=st.booleans(),
    )
    @mock.patch("train.main")
    def test_boolean_flags_propagate_to_main(
        self,
        mock_main,
        use_small,
        resume,
        optimizer_flags,
        lora,
        gpu,
        no_chunk,
        skip_checkpoint,
    ):
        deepspeed, adam8bit = optimizer_flags
        argv = list(BASE_ARGV)
        if use_small:
            argv.append("--use_small")
        if resume:
            argv.append("--resume")
        if deepspeed:
            argv.append("--deepspeed")
        if adam8bit:
            argv.append("--adam8bit")
        if lora:
            argv.append("--lora")
        if gpu:
            argv.append("--gpu")
        if no_chunk:
            argv.append("--no-chunk")
        if skip_checkpoint:
            argv.append("--skip_checkpoint")

        mock_main.reset_mock()
        cli_main(argv)

        mock_main.assert_called_once()
        _, kwargs = mock_main.call_args
        self.assertEqual(kwargs["use_deepspeed"], deepspeed)
        self.assertEqual(kwargs["use_adam8bit"], adam8bit)
        self.assertEqual(kwargs["use_lora"], lora)
        self.assertEqual(kwargs["use_gpu"], gpu)
        self.assertEqual(kwargs["use_chunk_and_check"], not no_chunk)
        self.assertEqual(kwargs["skip_checkpoint"], skip_checkpoint)

        args = mock_main.call_args.args
        self.assertEqual(args[0], Path(BASE_ARGV[1]))
        self.assertEqual(args[3], use_small)
        self.assertEqual(args[5], resume)


class RequiredArgumentTests(unittest.TestCase):
    @mock.patch("train.main")
    def test_missing_required_data_path_raises(self, mock_main):
        with self.assertRaises(SystemExit):
            cli_main([])
        mock_main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
