"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

This module contains constants used for the alpaka job generation.
"""

from typing import List, Dict, Union
import packaging.version
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

# alpaka sycl backend
# the alpaka cmake variables does not distinguish between the ONE api backends
# it uses other additional variables
ALPAKA_ACC_SYCL_ENABLE: str = "alpaka_ACC_SYCL_ENABLE"

# possible values of BUILD_TYPE
BUILD_TYPE: bashi.Parameter = "build_type"
CMAKE_RELEASE: int = 0
CMAKE_DEBUG: int = 1
CMAKE_RELEASE_WITH_DEBUG_INFO: int = 2
CMAKE_RELEASE_VER: ValueVersion = packaging.version.parse(str(CMAKE_RELEASE))
CMAKE_DEBUG_VER: ValueVersion = packaging.version.parse(str(CMAKE_DEBUG))
CMAKE_RELEASE_WITH_DEBUG_INFO_VER: ValueVersion = packaging.version.parse(
    str(CMAKE_RELEASE_WITH_DEBUG_INFO)
)
BUILD_TYPES: List[Union[str, int, float]] = [
    CMAKE_RELEASE,
    CMAKE_DEBUG,
    CMAKE_RELEASE_WITH_DEBUG_INFO,
]
BUILD_TYPES_NAMES: Dict[str, ValueVersion] = {
    "Release": CMAKE_RELEASE_VER,
    "Debug": CMAKE_DEBUG_VER,
    "RelWithDebInfo": CMAKE_RELEASE_WITH_DEBUG_INFO_VER,
}

# possible values of TEST_TYPE
JOB_EXECUTION_TYPE: bashi.Parameter = "job_execution_type"
JOB_EXECUTION_COMPILE_ONLY: int = 0
JOB_EXECUTION_RUNTIME: int = 1
JOB_EXECUTION_COMPILE_ONLY_VER: ValueVersion = packaging.version.parse(
    str(JOB_EXECUTION_COMPILE_ONLY)
)
JOB_EXECUTION_RUNTIME_VER: ValueVersion = packaging.version.parse(str(JOB_EXECUTION_RUNTIME))
JOB_EXECUTION_TYPES: List[Union[str, int, float]] = [
    JOB_EXECUTION_COMPILE_ONLY,
    JOB_EXECUTION_RUNTIME,
]
JOB_EXECUTION_TYPES_NAMES: Dict[str, ValueVersion] = {
    "compile_only": JOB_EXECUTION_COMPILE_ONLY_VER,
    "runtime": JOB_EXECUTION_RUNTIME_VER,
}

# enable mdspan support
MDSPAN: bashi.Parameter = "mdspan"

# CI pipeline
CI_PIPELINE_NAME: str = "stage_name"
CI_PIPELINE_COMPILE_ONLY: str = "compile_only"
CI_PIPELINE_COMPILE_ONLY_VER: ValueVersion = packaging.version.parse("0")
CI_PIPELINE_RUNTIME_CPU: str = "runtime_job_cpu"
CI_PIPELINE_RUNTIME_CPU_VER: ValueVersion = packaging.version.parse("1")
CI_PIPELINE_RUNTIME_GPU: str = "runtime_job_gpu"
CI_PIPELINE_RUNTIME_GPU_VER: ValueVersion = packaging.version.parse("2")
CI_PIPELINE_SPECIAL: str = "special_job"
CI_PIPELINE_SPECIAL_VER: ValueVersion = packaging.version.parse("3")

CI_PIPELINE_NAME_MAPPING: Dict[str, ValueVersion] = {
    CI_PIPELINE_COMPILE_ONLY: CI_PIPELINE_COMPILE_ONLY_VER,
    CI_PIPELINE_RUNTIME_CPU: CI_PIPELINE_RUNTIME_CPU_VER,
    CI_PIPELINE_RUNTIME_GPU: CI_PIPELINE_RUNTIME_GPU_VER,
    CI_PIPELINE_SPECIAL: CI_PIPELINE_SPECIAL_VER,
}


def get_version_aliases() -> Dict[bashi.ValueName, Dict[bashi.ValueVersion, str]]:
    """Return a list of value-version aliases which can be set for print_row_nice()

    Returns:
        Dict[bashi.ValueName, Dict[bashi.ValueVersion, str]]: _description_
    """
    version_aliases = {}
    for val_name, version_map in [
        (BUILD_TYPE, BUILD_TYPES_NAMES),
        (JOB_EXECUTION_TYPE, JOB_EXECUTION_TYPES_NAMES),
        (CI_PIPELINE_NAME, CI_PIPELINE_NAME_MAPPING),
    ]:
        version_map_parsed: Dict[bashi.ValueVersion, str] = {}
        for alias, ver in version_map.items():
            version_map_parsed[ver] = alias
        version_aliases[val_name] = version_map_parsed

    return version_aliases


RT_HOST_COMPILER_CUDA_SUPPORT: str = "rt_host_compiler_cuda_support"
RT_CLANG_CUDA_MAX_CUDA_SUPPORT: str = "rt_clang_cuda_max_cuda_support"
