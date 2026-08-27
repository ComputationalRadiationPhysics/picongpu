"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Add the CI pipeline to the combinations.
"""

from copy import deepcopy
from typeguard import typechecked
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import (
    CI_PIPELINE_NAME,
    CI_PIPELINE_COMPILE_ONLY_VER,
    CI_PIPELINE_RUNTIME_CPU_VER,
    CI_PIPELINE_RUNTIME_GPU_VER,
    JOB_EXECUTION_TYPE,
    JOB_EXECUTION_COMPILE_ONLY_VER,
    JOB_EXECUTION_RUNTIME_VER,
)


# pylint: disable=too-many-branches
@typechecked
def add_ci_pipeline_type(combination_list: bashi.CombinationList) -> bashi.CombinationList:
    """Add the name of the CI pipeline."""
    combination_list_copy = deepcopy(combination_list)

    for comb in combination_list_copy:
        # GCC or Clang as CPU compiler
        if (
            comb[DEVICE_COMPILER].name in (GCC, CLANG)
            and comb[HOST_COMPILER].name in (GCC, CLANG)
            and comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_RUNTIME_VER
        ):
            comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                CI_PIPELINE_NAME, CI_PIPELINE_RUNTIME_CPU_VER
            )

        # Hipcc compiler
        if comb[DEVICE_COMPILER].name == HIPCC:
            if comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_COMPILE_ONLY_VER:
                comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                    CI_PIPELINE_NAME, CI_PIPELINE_COMPILE_ONLY_VER
                )
            if comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_RUNTIME_VER:
                comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                    CI_PIPELINE_NAME, CI_PIPELINE_RUNTIME_GPU_VER
                )

        # ICPX
        if comb[DEVICE_COMPILER].name == ICPX:
            if (
                comb[ALPAKA_ACC_ONEAPI_CPU_ENABLE].version == ON_VER
                and comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_RUNTIME_VER
            ):
                comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                    CI_PIPELINE_NAME, CI_PIPELINE_RUNTIME_CPU_VER
                )
            for one_api_backend in (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ALPAKA_ACC_ONEAPI_FPGA_ENABLE):
                if (
                    comb[one_api_backend].version == ON_VER
                    and comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_COMPILE_ONLY_VER
                ):
                    comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                        CI_PIPELINE_NAME, CI_PIPELINE_COMPILE_ONLY_VER
                    )

        # CUDA
        for device_compiler in (NVCC, CLANG_CUDA):
            if comb[DEVICE_COMPILER].name == device_compiler:
                if comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_COMPILE_ONLY_VER:
                    comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                        CI_PIPELINE_NAME, CI_PIPELINE_COMPILE_ONLY_VER
                    )
                if comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_RUNTIME_VER:
                    comb[CI_PIPELINE_NAME] = bashi.ParameterValue(
                        CI_PIPELINE_NAME, CI_PIPELINE_RUNTIME_GPU_VER
                    )

    return combination_list_copy
