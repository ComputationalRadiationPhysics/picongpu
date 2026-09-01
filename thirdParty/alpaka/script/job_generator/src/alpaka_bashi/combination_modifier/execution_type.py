"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Add the job execution type to the combinations.
"""

from copy import deepcopy
from typing import Dict, List
import packaging.version
from typeguard import typechecked
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import (
    JOB_EXECUTION_TYPE,
    JOB_EXECUTION_COMPILE_ONLY_VER,
    JOB_EXECUTION_RUNTIME_VER,
)
import alpaka_bashi.versions


@typechecked
def execution_type_device_compiler_gcc_and_clang(
    combination_list: bashi.CombinationList,
) -> bashi.CombinationList:
    """Annotate all jobs where gcc or clang are the device compiler as runtime job."""
    combination_list_copy = deepcopy(combination_list)

    for comb in combination_list_copy:
        if comb[DEVICE_COMPILER].name in (GCC, CLANG) and comb[HOST_COMPILER].name in (GCC, CLANG):
            if JOB_EXECUTION_TYPE in comb:
                raise RecursionError(
                    "JOB_EXECUTION_TYPE is already defined in the combinations: "
                    f"{bashi.get_str_row_nice(comb)}"
                )
            comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME_VER
            )

    return combination_list_copy


@typechecked
def execution_type_hipcc(combination_list: bashi.CombinationList) -> bashi.CombinationList:
    """Annotate for each hipcc version one job as runtime. The rest is compile time."""
    combination_list_copy = deepcopy(combination_list)

    hipcc_versions = [
        packaging.version.parse(str(ver))
        for ver in alpaka_bashi.versions.get_software_versions_for_alpaka()[HIPCC]
    ]
    for comb in combination_list_copy:
        if comb[DEVICE_COMPILER].name == HIPCC:
            if JOB_EXECUTION_TYPE in comb:
                raise RecursionError(
                    "JOB_EXECUTION_TYPE is already defined in the combinations: "
                    f"{bashi.get_str_row_nice(comb)}"
                )
            if comb[DEVICE_COMPILER].version in hipcc_versions:
                comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                    JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME_VER
                )
                hipcc_versions.remove(comb[DEVICE_COMPILER].version)
            else:
                comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                    JOB_EXECUTION_TYPE, JOB_EXECUTION_COMPILE_ONLY_VER
                )

    return combination_list_copy


@typechecked
def execution_type_icpx(combination_list: bashi.CombinationList) -> bashi.CombinationList:
    """Annotate all jobs with the OneAPI CPU backend as runtime job.
    Annotate all jobs with the OneAPI GPU and FPGA backend as compile only job."""
    combination_list_copy = deepcopy(combination_list)

    for comb in combination_list_copy:
        if comb[DEVICE_COMPILER].name == ICPX:
            if JOB_EXECUTION_TYPE in comb:
                raise RecursionError(
                    "JOB_EXECUTION_TYPE is already defined in the combinations: "
                    f"{bashi.get_str_row_nice(comb)}"
                )
            if comb[ALPAKA_ACC_ONEAPI_CPU_ENABLE].version == ON_VER:
                comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                    JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME_VER
                )
            for sycl_backend in (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ALPAKA_ACC_ONEAPI_FPGA_ENABLE):
                if comb[sycl_backend].version == ON_VER:
                    comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                        JOB_EXECUTION_TYPE, JOB_EXECUTION_COMPILE_ONLY_VER
                    )

    return combination_list_copy


@typechecked
def find_latest_cuda_sdk_minor_versions(
    combination_list: bashi.CombinationList,
) -> Dict[ValueName, List[ValueVersion]]:
    """For the compiler Nvcc+GCC, Nvcc+Clang and Clang-CUDA, find for each major release the newest
    and oldest minor CUDA SDK version, named in the combintation-list.

    Args:
        combination_list (bashi.CombinationList): combination-list

    Raises:
        RuntimeError: If there is a combination with an CUDA compiler and an disabled CUDA backend.

    Returns:
        Dict[ValueName, List[ValueVersion]]: The key is the name of the host compiler. The value is
        a list of found CUDA SDK versions.
    """
    versions: Dict[ValueName, List[ValueVersion]] = {GCC: [], CLANG: [], CLANG_CUDA: []}

    # {"compiler name" : { "CUDA major version" : []"min version", "max version"] } }
    tmp_versions: Dict[ValueName, Dict[int, List[ValueVersion]]] = {
        GCC: {},
        CLANG: {},
        CLANG_CUDA: {},
    }
    # find the oldest and newest CUDA SDK version for each CUDA major version and compiler
    for comb in combination_list:
        for host_compiler, device_compiler in (
            (GCC, NVCC),
            (CLANG, NVCC),
            (CLANG_CUDA, CLANG_CUDA),
        ):
            if (
                comb[HOST_COMPILER].name == host_compiler
                and comb[DEVICE_COMPILER].name == device_compiler
            ):
                major_version = comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version.major
                if major_version == 0:
                    raise RuntimeError("Major version 0 should be not possible")
                if major_version in tmp_versions[host_compiler]:
                    tmp_versions[host_compiler][major_version][0] = min(
                        comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version,
                        tmp_versions[host_compiler][major_version][0],
                    )
                    tmp_versions[host_compiler][major_version][1] = max(
                        comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version,
                        tmp_versions[host_compiler][major_version][1],
                    )
                else:
                    tmp_versions[host_compiler][major_version] = [
                        comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version,
                        comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version,
                    ]

    for compiler_name, major_versions in tmp_versions.items():
        for cuda_versions in major_versions.values():
            for cuda_version in cuda_versions:
                versions[compiler_name].append(cuda_version)

    # make CUDA version unique
    for compiler_name in versions:
        versions[compiler_name] = list(set(versions[compiler_name]))

    return versions


@typechecked
def execution_type_cuda_backend(combination_list: bashi.CombinationList) -> bashi.CombinationList:
    """Annotate for each cuda version one job as runtime. The rest is compile time."""
    combination_list_copy = deepcopy(combination_list)

    cuda_versions = find_latest_cuda_sdk_minor_versions(combination_list_copy)
    for comb in combination_list_copy:
        for host_compiler, device_compiler in (
            (GCC, NVCC),
            (CLANG, NVCC),
            (CLANG_CUDA, CLANG_CUDA),
        ):
            if (
                comb[HOST_COMPILER].name == host_compiler
                and comb[DEVICE_COMPILER].name == device_compiler
            ):
                if JOB_EXECUTION_TYPE in comb:
                    raise RecursionError(
                        "JOB_EXECUTION_TYPE is already defined in the combinations: "
                        f"{bashi.get_str_row_nice(comb)}"
                    )

                if comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version in cuda_versions[host_compiler]:
                    comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                        JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME_VER
                    )
                    cuda_versions[host_compiler].remove(comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version)
                else:
                    comb[JOB_EXECUTION_TYPE] = bashi.ParameterValue(
                        JOB_EXECUTION_TYPE, JOB_EXECUTION_COMPILE_ONLY_VER
                    )

    return combination_list_copy
