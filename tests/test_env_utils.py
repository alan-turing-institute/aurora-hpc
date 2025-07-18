import unittest

from aurora_hpc.env_utils import parse_nodelist


class TestEnvUtils(unittest.TestCase):
    def test_parse_hyphen(self) -> None:
        """Parses a range"""
        self.assertListEqual(
            ["partition-p-1", "partition-p-2"],
            parse_nodelist("partition-p-[1-2]"),
        )

    def test_parse_comma(self) -> None:
        """Parses a range"""
        self.assertListEqual(
            ["partition-p-1", "partition-p-3", "partition-p-5"],
            parse_nodelist("partition-p-[1,3,5]"),
        )

    def test_raises_invalid_nodelist(self) -> None:
        """Raises an error for non-] final char."""
        with self.assertRaises(ValueError):
            parse_nodelist("partition-p-[1-2]-")

    def test_raises_no_match(self) -> None:
        """Raises an error for no match."""
        with self.assertRaises(ValueError):
            parse_nodelist("[]")


if __name__ == "__main__":
    unittest.main()
