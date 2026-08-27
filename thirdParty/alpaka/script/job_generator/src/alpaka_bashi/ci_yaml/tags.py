"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the tags of the GitLab CI test job yaml.
"""

from typing import Dict, Any
from typeguard import typechecked
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import (
    JOB_EXECUTION_TYPE,
    JOB_EXECUTION_COMPILE_ONLY_VER,
    JOB_EXECUTION_RUNTIME_VER,
)


@typechecked
def set_tags(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set the tags of the GitLab CI test job yaml depending on the combination.
    The tags decide which CI runner is used, e.g. the CPU runner for compile only jobs
    or the Nvidia runner for CUDA runtime jobs

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    if combination[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_COMPILE_ONLY_VER:
        job_body["tags"] = ["x86_64", "cpuonly"]
        return

    if combination[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_RUNTIME_VER:
        if combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            if combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version < packaging.version.parse("13.0"):
                job_body["tags"] = ["x86_64", "cuda"]
                return
            # with CUDA 13.0 we have to use the Nvidia A100, because the Nvidia Quadro P5000 is not
            # supported anymore
            job_body["tags"] = ["x86_64", "cuda", "a100"]
            return
        if combination[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
            job_body["tags"] = ["x86_64", "rocm"]
            return

        # cpu runtime jobs
        job_body["tags"] = ["x86_64", "cpuonly"]
        return

    # fallback
    job_body["tags"] = ["x86_64", "cpuonly"]
