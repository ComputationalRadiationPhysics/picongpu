"""Copyright 2023 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Functions to modify order of the job list.
"""

from typing import List, Dict, Tuple

from typeguard import typechecked

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


@typechecked
def reorder_jobs(job_matrix: List[Dict[str, Tuple[str, str]]]):
    """Vikunja specific function, to move jobs, which matches certain properties to the first waves.

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): The job_matrix.
    """
    pass
