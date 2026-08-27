"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate CI job names depending on the value-versions of a combination.
"""

from typeguard import typechecked
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import (
    MDSPAN,
    BUILD_TYPE,
    CMAKE_RELEASE_VER,
    CMAKE_DEBUG_VER,
    JOB_EXECUTION_TYPE,
    JOB_EXECUTION_COMPILE_ONLY_VER,
)


@typechecked
def get_job_suffix(combination: bashi.Combination) -> str:
    """Generate a suffix for a CI job name depending on the value-versions of a combination.

    Args:
        comb (Dict[str, Tuple[str, str]]): combination.

    Returns:
        str: name suffix.
    """
    version_str = ""
    for software in [CMAKE, UBUNTU, CXX_STANDARD, MDSPAN, BUILD_TYPE]:
        if software in combination:
            if combination[software].name == CXX_STANDARD:
                version_str += f"_cxx{str(combination[software].version)}"
            elif combination[software].name == UBUNTU:
                version_str += (
                    f"_{combination[software].name}"
                    f"{bashi.ubuntu_version_to_string(combination[software].version)}"
                )
            elif combination[software].name == BUILD_TYPE:
                continue
            elif combination[software].name == MDSPAN:
                if combination[software].version == ON_VER:
                    version_str += "_mdspan"
            else:
                version_str += f"_{combination[software].name}{str(combination[software].version)}"

    # make sure, that the build type is at the end of the name
    if combination[BUILD_TYPE].version == CMAKE_RELEASE_VER:
        version_str += "_release"
    if combination[BUILD_TYPE].version == CMAKE_DEBUG_VER:
        version_str += "_debug"

    return version_str


@typechecked
def get_job_name(comb: bashi.Combination) -> str:
    """Generate a CI job name depending on the value-versions of a combination.

    Args:
        comb (Dict[str, Tuple[str, str]]): combination.

    Returns:
        str: name suffix.
    """
    # the job name starts with the device compiler
    job_name = f"linux_{comb[DEVICE_COMPILER].name}{str(comb[DEVICE_COMPILER].version)}"
    # if the nvcc is the device compiler, add also the host compiler to the name
    if comb[DEVICE_COMPILER].name == NVCC:
        job_name = job_name + f"-{comb[HOST_COMPILER].name}{str(comb[HOST_COMPILER].version)}"
    # if Clang-CUDA is the device compiler, add also the CUDA SDK version to the name
    if comb[DEVICE_COMPILER].name == CLANG_CUDA:
        job_name = job_name + f"-cuda{str(comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version)}"

    # if the OneAPI backend is used, specify for which device type.
    for oneapi_backend, oneapi_suffix in (
        (ALPAKA_ACC_ONEAPI_CPU_ENABLE, "-cpu"),
        (ALPAKA_ACC_ONEAPI_GPU_ENABLE, "-gpu"),
        (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, "-fpga"),
    ):
        if comb[oneapi_backend].version == ON_VER:
            job_name += oneapi_suffix

    if comb[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_COMPILE_ONLY_VER:
        job_name += "_compile_only"

    return job_name + get_job_suffix(comb)
