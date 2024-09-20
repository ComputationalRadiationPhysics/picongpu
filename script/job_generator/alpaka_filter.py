"""Copyright 2023 Simeon Ehrig, Jan Stephan
SPDX-License-Identifier: MPL-2.0

Alpaka project specific filter rules.
"""

from typing import List

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    is_in_row,
    row_check_backend_version,
)


def alpaka_post_filter(row: List) -> bool:
    # debug builds with clang 15 and 16 as CUDA compiler produce a compiler error
    # see here: https://github.com/llvm/llvm-project/issues/58491
    if (
        is_in_row(row, BUILD_TYPE)
        and row[param_map[BUILD_TYPE]][VERSION] == CMAKE_DEBUG
        and row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA)
    ):
        for clang_cuda_version in ["15", "16", "17"]:
            if row_check_version(row, HOST_COMPILER, "==", clang_cuda_version):
                return False

    # Debug builds with nvcc <= 11.6 produce compiler errors
    if (
        is_in_row(row, BUILD_TYPE)
        and row[param_map[BUILD_TYPE]][VERSION] == CMAKE_DEBUG
        and row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, DEVICE_COMPILER, "<=", "11.6")
    ):
        return False

    # because of a compiler bug, we disable mdspan for NVCC <= 11.2
    if (
        row_check_version(row, MDSPAN, "==", ON_VER)
        and row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, DEVICE_COMPILER, "<=", "11.2")
    ):
        return False

    # OpenMP is not supported for clang as cuda compiler
    # https://github.com/alpaka-group/alpaka/issues/639
    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA) and (
        row_check_backend_version(row, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, "==", ON_VER)
        or row_check_backend_version(
            row, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, "==", ON_VER
        )
    ):
        return False

    # there is a compiler bug in GCC 11.4 which avoids working with NVCC 11.5
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and (
            row_check_version(row, DEVICE_COMPILER, "==", "11.4")
            or row_check_version(row, DEVICE_COMPILER, "==", "11.5")
        )
        and row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "==", "11")
    ):
        return False

    # cmake 3.24 and older does not support C++20 for nvcc
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, CXX_STANDARD, ">=", "20")
        and row_check_version(row, CMAKE, "<", "3.25")
    ):
        return False

    # Debug builds with HIP/ROCm 6.2 produce compiler errors
    if (
        is_in_row(row, BUILD_TYPE)
        and row[param_map[BUILD_TYPE]][VERSION] == CMAKE_DEBUG
        and row_check_name(row, DEVICE_COMPILER, "==", HIPCC)
        and row_check_version(row, DEVICE_COMPILER, "==", "6.2")
    ):
        return False

     # g++-12 is not available on the Ubuntu 20.04 ppa's
    if (
        row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "==", "12")
        and row_check_version(row, UBUNTU, "==", "20.04")
    ):
        return False

    # there is a bug with g++-13 and cuda 12.4 on Ubuntu 20.04
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, DEVICE_COMPILER, "==", "12.4")
        and row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "==", "13")
        and row_check_version(row, UBUNTU, "==", "20.04")
    ):
        return False

    return True
