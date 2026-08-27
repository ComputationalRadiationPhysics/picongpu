"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

alpaka_bashi package
"""

from alpaka_bashi.versions import (
    get_software_versions_for_alpaka,
    get_version_aliases,
    get_alpaka_version_relation,
)
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.alpaka_filter import AlpakaFilter
from alpaka_bashi.verify import verify
from alpaka_bashi.runtime_info import get_runtime_infos
from alpaka_bashi.combination import add_combinations_parameters
from alpaka_bashi.pipeline import distribute_to_pipelines
from alpaka_bashi.ci_yaml.writer import get_job_configuration, get_dummy_job_yaml, write_job_yaml
from alpaka_bashi.combination_modifier.filter import filter_combinations
from alpaka_bashi.special_jobs.writer import get_special_jobs
