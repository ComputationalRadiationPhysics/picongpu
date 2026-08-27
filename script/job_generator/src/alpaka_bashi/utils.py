"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Utils for the job-generator
"""

import sys
from typeguard import typechecked
import termcolor


@typechecked
def print_warn(msg: str):
    """Print message in yellow with a [WARNING] prefix.

    Args:
        msg (str): warning text
    """
    print(termcolor.colored(f"[WARNING]: {msg}", "yellow"), file=sys.stderr)
