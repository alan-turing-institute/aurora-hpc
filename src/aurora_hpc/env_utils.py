"""Utilities for dealing with HPC environment variables."""

import re


def parse_nodelist(nodelist: str) -> list[str]:
    """Parses a Slurm nodelist string into a list of node names."""

    if nodelist[-1] != "]":
        raise ValueError("Nodelist must end with a closing square bracket.")

    open_bracket = nodelist.index("[")
    prefix = nodelist[0:open_bracket]
    node_range = nodelist[open_bracket:]

    # Match patterns like "partition-p-[1-3]"
    hyphen_pattern = r"(\d+)-(\d+)"

    if match := re.search(hyphen_pattern, node_range):
        start, end = int(match.group(1)), int(match.group(2))
        return [f"{prefix}{i}" for i in range(start, end + 1)]

    # Match patterns like "partition-p-[1,3,5]"
    comma_pattern = r"(\d+)(?:,(\d+))*"
    if match := re.search(comma_pattern, node_range):
        return [f"{prefix}{i}" for i in match.group(0).split(",")]

    raise ValueError("Nodelist does not match expected patterns.")
