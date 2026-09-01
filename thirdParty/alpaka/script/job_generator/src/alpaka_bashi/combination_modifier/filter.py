"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Filter combinations.
"""

import re
import bashi
from alpaka_bashi.ci_yaml.names import get_job_name


def filter_combinations(
    combination_list: bashi.CombinationList, job_name_regex: str
) -> bashi.CombinationList:
    """Filter combinations by a given regex. The job names are temporary generated. If a job name
    does not match the job_name_regex, it will be remove.

    Args:
        combination_list (bashi.CombinationList): Combination List
        job_name_regex (str): job name regex

    Returns:
        bashi.CombinationList: filtered combination list
    """
    compiled_regex = re.compile(job_name_regex)

    return [
        combination
        for combination in combination_list
        if compiled_regex.match(get_job_name(combination))
    ]
