"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verify generated combinations.
"""

from typing import Dict, Callable, List, cast
from alpaka_bashi.versions import (
    get_allowed_backend_combinations,
    get_used_backends,
    get_used_compiler_versions,
)
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.runtime_info
import alpaka_bashi.globals
from alpaka_bashi.runtime_info import ClangCUDAMaxSupportsCuda


def remove_disabled_serial_backend(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """All combinations requires an enabled serial backend. Therefore remove all pairs with disabled
    serial backend.
    """
    bashi.remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        value_version1=OFF,
    )


def remove_disabled_backend_for_compiler(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all combination where a specific backend cannot be disabled for a given host or device
    compiler"""
    backend_compilers = {
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE: [
            (HOST_COMPILER, GCC),
            (DEVICE_COMPILER, GCC),
            (HOST_COMPILER, CLANG),
            (DEVICE_COMPILER, CLANG),
            (DEVICE_COMPILER, NVCC),
        ],
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: [
            (HOST_COMPILER, GCC),
            (DEVICE_COMPILER, GCC),
            (HOST_COMPILER, CLANG),
            (DEVICE_COMPILER, CLANG),
            (DEVICE_COMPILER, NVCC),
        ],
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE: [
            (HOST_COMPILER, GCC),
            (DEVICE_COMPILER, GCC),
            (HOST_COMPILER, CLANG),
            (DEVICE_COMPILER, CLANG),
            (HOST_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, NVCC),
        ],
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: [
            # only device compiler GCC and Clang because depending if GCC can be used as NVCC
            # compiler or not it is possible that the this compilers are tested without TBB
            (DEVICE_COMPILER, GCC),
            (DEVICE_COMPILER, CLANG),
            (HOST_COMPILER, ICPX),
            (DEVICE_COMPILER, ICPX),
        ],
    }

    for backend, compilers in backend_compilers.items():
        for parameter, value_name in compilers:
            bashi.remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=backend,
                value_min_version1=OFF,
                value_max_version1=OFF,
                parameter2=parameter,
                value_name2=value_name,
            )


def remove_simple_backend_backend_combinations(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all invalid combinations of the enabled/disabled backends."""
    all_one_api_backends = "all_one_api_backends"
    for (
        backend1,
        state1,
        backend2,
        state2,
    ) in [
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
    ]:
        if backend2 == all_one_api_backends:
            second_backends = ONE_API_BACKENDS
        else:
            second_backends = [backend2]
        for second_backend in second_backends:
            bashi.remove_parameter_value_pairs(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=backend1,
                value_version1=state1,
                parameter2=second_backend,
                value_version2=state2,
            )


def remove_cuda_backend_backend_combinations(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all invalid combinations of an enabled CUDA backend and a enabled/disabled other
    backends. Handles the special case, the CUDA backend has a version number instead is simply
    enabled."""
    for backend, state in [
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF),
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
    ]:
        bashi.remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=backend,
            value_min_version1=state,
            value_max_version1=state,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_min_version2=OFF,
            value_min_version2_inclusive=False,
        )


def remove_non_used_nvcc_device_compiler(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
    run_infos: Dict[str, Callable[..., bool]],
):
    """Removes the the combination of disabled TBB backend and host compiler GCC or Clang.
    The TBB backend can be only disabled, if the GCC or Clang host compiler can be used as host
    compiler for nvcc. The enabled TBB backend is tested with all Clang and GCC versions when GCC or
    Clang is the device compiler and therefore also the host compiler."""
    if alpaka_bashi.globals.RT_HOST_COMPILER_CUDA_SUPPORT in run_infos:
        host_compiler_supports_cuda = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            run_infos[alpaka_bashi.globals.RT_HOST_COMPILER_CUDA_SUPPORT],
        )

        for compiler in (GCC, CLANG):
            bashi.remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=compiler,
                value_min_version1=str(host_compiler_supports_cuda.get_max_version(compiler)),
                value_min_version1_inclusive=False,
                parameter2=ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
                value_min_version2=OFF,
                value_max_version2=OFF,
            )


def remove_hip_62_debug_build(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove combination of Hipcc and CMake debug build"""
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        bashi.remove_parameter_value_pairs(
            parameter_value_pairs=parameter_value_pairs,
            removed_parameter_value_pairs=removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            value_version1=6.2,
            parameter2=alpaka_bashi.globals.BUILD_TYPE,
            value_version2=alpaka_bashi.globals.CMAKE_DEBUG,
        )


def remove_ubuntu2204(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Except for some HIP versions and Clang 16 and older, Ubuntu 22.04 should be not used.
    See bashi.UBUNTU_HIP_VERSION_RANGE"""
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        for compiler in (GCC, ICPX):
            bashi.remove_parameter_value_pairs(
                parameter_value_pairs=parameter_value_pairs,
                removed_parameter_value_pairs=removed_parameter_value_pairs,
                parameter1=compiler_type,
                value_name1=compiler,
                parameter2=UBUNTU,
                value_version2="22.04",
            )

        bashi.remove_parameter_value_pairs_ranges(
            parameter_value_pairs=parameter_value_pairs,
            removed_parameter_value_pairs=removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=CLANG,
            value_min_version1="17",
            parameter2=UBUNTU,
            value_max_version2="22.04",
        )
    for backend, version in [
        (ALPAKA_ACC_ONEAPI_CPU_ENABLE, ON),
        (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON),
        (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON),
    ]:
        bashi.remove_parameter_value_pairs(
            parameter_value_pairs=parameter_value_pairs,
            removed_parameter_value_pairs=removed_parameter_value_pairs,
            parameter1=backend,
            value_version1=version,
            parameter2=UBUNTU,
            value_version2="22.04",
        )


def remove_clang16_and_older_cuda(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all combinations affecting Clang 16 and older and CUDA."""
    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs=parameter_value_pairs,
        removed_parameter_value_pairs=removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        value_max_version1="16",
        parameter2=DEVICE_COMPILER,
        value_name2=NVCC,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
    )

    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs=parameter_value_pairs,
        removed_parameter_value_pairs=removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        value_max_version1="16",
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
    )

    # remove disable TBB backend. Only if Clang is used as CUDA host compiler, TBB will be disabled.
    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs=parameter_value_pairs,
        removed_parameter_value_pairs=removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        value_max_version1="16",
        parameter2=ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
        value_max_version2=OFF,
        value_min_version2_inclusive=False,
    )


def remove_ubuntu2404(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove the combination of Clang 16 and older and Ubuntu 24.04."""
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):

        bashi.remove_parameter_value_pairs_ranges(
            parameter_value_pairs=parameter_value_pairs,
            removed_parameter_value_pairs=removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=CLANG,
            value_max_version1="16",
            parameter2=UBUNTU,
            value_min_version2="24.04",
        )


def remove_clang_cuda_not_used_backend_combinations(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
    run_infos: Dict[str, Callable[..., bool]],
):
    """This backend pairs are removed, because a CUDA SDK version cannot be used by the available
    Clang-CUDA versions."""
    if alpaka_bashi.globals.RT_CLANG_CUDA_MAX_CUDA_SUPPORT in run_infos:
        max_cuda_version = cast(
            ClangCUDAMaxSupportsCuda, run_infos[alpaka_bashi.globals.RT_CLANG_CUDA_MAX_CUDA_SUPPORT]
        ).max_cuda_sdk_version

        # the OpenMP backends can be only enabled with the nvcc
        for other_backend in (
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
        ):
            bashi.remove_parameter_value_pairs_ranges(
                parameter_value_pairs=parameter_value_pairs,
                removed_parameter_value_pairs=removed_parameter_value_pairs,
                parameter1=ALPAKA_ACC_GPU_CUDA_ENABLE,
                value_min_version1=str(max_cuda_version),
                value_min_version1_inclusive=False,
                parameter2=other_backend,
                value_min_version2=OFF,
                value_min_version2_inclusive=True,
                value_max_version2=OFF,
                value_max_version2_inclusive=True,
            )


def verify(
    combination_list: bashi.CombinationList,
    param_value_matrix: bashi.ParameterValueMatrix,
    version_relation: bashi.VersionRelation,
    run_infos: Dict[str, Callable[..., bool]],
) -> bool:
    """Check if all expected parameter-value-pairs exists in the combination-list.

    Args:
        combination_list (CombinationList): The generated combination list.
        param_value_matrix (ParameterValueMatrix): The expected parameter-values-pairs are generated
            from the parameter-value-list.

    Returns:
        bool: True if it found all pairs
    """

    expected_param_val_tuple, unexpected_param_val_tuple = (
        bashi.get_expected_bashi_parameter_value_pairs(
            param_value_matrix, version_relation, run_infos
        )
    )

    remove_disabled_serial_backend(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_disabled_backend_for_compiler(expected_param_val_tuple, unexpected_param_val_tuple)
    bashi.remove_unsupported_compiler_backend_combinations(
        expected_param_val_tuple,
        unexpected_param_val_tuple,
        list(get_used_compiler_versions().keys()),
        get_used_backends(),
        get_allowed_backend_combinations(),
    )
    remove_simple_backend_backend_combinations(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_cuda_backend_backend_combinations(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_non_used_nvcc_device_compiler(
        expected_param_val_tuple, unexpected_param_val_tuple, run_infos
    )
    remove_hip_62_debug_build(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_ubuntu2204(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_clang16_and_older_cuda(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_ubuntu2404(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_clang_cuda_not_used_backend_combinations(
        expected_param_val_tuple, unexpected_param_val_tuple, run_infos
    )

    expected_param_val_okay = bashi.check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    )
    unexpected_param_val_okay = bashi.check_unexpected_parameter_value_pair_in_combination_list(
        combination_list, unexpected_param_val_tuple
    )

    return expected_param_val_okay and unexpected_param_val_okay
