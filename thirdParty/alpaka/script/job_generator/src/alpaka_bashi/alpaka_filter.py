"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

from typing import Dict, Callable, IO, List
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.versions import get_used_backends, get_allowed_backend_combinations


def only_cuda_compiler_backends(combinations: List[bashi.CompilerBackendCombination]) -> bool:
    """Return True, if there are only bashi.CompilerBackendCombination in the list, which contains
    the CUDA compilers and backend."""
    for comb in combinations:
        if ALPAKA_ACC_GPU_CUDA_ENABLE not in comb.backends:
            return False
    return True


def only_clang_cuda_compiler_backends(combinations: List[bashi.CompilerBackendCombination]) -> bool:
    """Return True, if only compiler backend combinations with the Clang-CUDA compiler exist."""
    for comb in combinations:
        host_compiler = comb[0]
        if host_compiler != CLANG_CUDA:
            return False
    return True


def check_only_valid_backend_combinations_a1(
    row: bashi.BashiRow, alpaka_filter: "AlpakaFilter"
) -> bool:
    """
    Check if still possible valid backend combinations exist.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if (
        len(
            bashi.get_valid_compiler_backend_combinations(
                row, get_allowed_backend_combinations(), get_used_backends()
            )
        )
        == 0
    ):
        alpaka_filter.reason("No valid backend combination available.")
        return False
    return True


def check_cuda_sdk_host_compiler_a2(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Check if there is a GCC or Clang version which supports the CUDA SDK version in the row.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if only_cuda_compiler_backends(
        bashi.get_valid_compiler_backend_combinations(
            row, get_allowed_backend_combinations(), get_used_backends()
        )
    ):
        if (
            row[HOST_COMPILER].name in (GCC, CLANG)
            and RT_HOST_COMPILER_CUDA_SUPPORT in alpaka_filter.runtime_infos
            and not alpaka_filter.runtime_infos[RT_HOST_COMPILER_CUDA_SUPPORT](
                row[HOST_COMPILER].name, row[HOST_COMPILER].version
            )
        ):
            alpaka_filter.reason(
                "Only backend combinations with CUDA backend possible. There is no CUDA SDK "
                f"version, which supports the host compiler {row[HOST_COMPILER].name}-"
                f"{row[HOST_COMPILER].version}"
            )
            return False
    return True


def check_debug_build_hip_a3(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    HIP/ROCm 6.2 does not support debug builds because of a bug.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if row[BUILD_TYPE].version == CMAKE_DEBUG_VER:
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if row[compiler_type].name == HIPCC and row[
                compiler_type
            ].version == packaging.version.parse("6.2"):
                alpaka_filter.reason("Debug builds with HIP/ROCm 6.2 produce compiler errors.")
                return False
    return True


def check_ubuntu_22_04_specifics_a4(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Check some conditions which are related to Ubuntu 22.04
    - Only HIPCC and Clang will be tested on Ubuntu 22.04.
    - Clang 17 and later will be tested on Ubuntu 24.04 and later.
    - The CUDA and OneAPI backend will be not tested on Ubuntu 22.04.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    # OVERWORK: remove/overwork me, if bashi supports standard library
    if row[UBUNTU].version < packaging.version.parse("24.04"):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name not in (CLANG, HIPCC):
                alpaka_filter.reason("Only HIPCC and Clang will be tested on Ubuntu 22.04")
                return False

            if row[compiler_type].name == CLANG and row[
                compiler_type
            ].version > packaging.version.parse("16"):
                alpaka_filter.reason("Clang 17 and later will be tested on Ubuntu 24.04 and later.")
                return False

        for backend in ONE_API_BACKENDS + [ALPAKA_ACC_GPU_CUDA_ENABLE]:
            if row[backend].version != OFF_VER:
                alpaka_filter.reason(
                    f"The backend {row[backend].name} will be not used on Ubuntu 22.04 and "
                    "older."
                )
                return False
    return True


def check_clang_16_and_older_a5(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Check some conditions which are related to Clang 16 and older
    - Clang 16 and older does not support the libc++-13 of Ubuntu 24.04 host compiler GCC 13.
    - If a specific version of NVCC and Clang can both be installed on Ubuntu 24.04.
    - If clang is used, at least one CPU backend needs to be enabled.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    # OVERWORK: remove/overwork me, if bashi supports standard library
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        if row[compiler_type].name == CLANG and row[
            compiler_type
        ].version <= packaging.version.parse("16"):
            if row[UBUNTU].version >= packaging.version.parse("24.04"):
                alpaka_filter.reason(
                    f"Clang {row[compiler_type].version} does not support libc++-13 and later "
                    f"of the host compiler of Ubuntu {row[UBUNTU].version}"
                )
                return False

            if row[DEVICE_COMPILER].name == NVCC and row[
                DEVICE_COMPILER
            ].version >= packaging.version.parse("12.0"):
                alpaka_filter.reason(
                    f"NVCC {row[DEVICE_COMPILER].version} is only available on UBUNTU 24.04 "
                    f"and later but Clang {row[HOST_COMPILER].version} does not support 24.04 "
                    "and later."
                )
                return False

            if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= packaging.version.parse("12.0"):
                alpaka_filter.reason(
                    f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} is only available on "
                    f"UBUNTU 24.04 and later but Clang {row[HOST_COMPILER].version} does not "
                    "support 24.04 and later."
                )
                return False

            for cpu_backend in CPU_BACKENDS:
                if row[cpu_backend].version == OFF_VER:
                    alpaka_filter.reason(
                        f"Clang {row[compiler_type].version} works only together with CPU "
                        "backends."
                    )
                    return False
    return True


def check_existing_clang_cuda_for_cuda_sdk_version_a6(
    row: bashi.BashiRow, alpaka_filter: "AlpakaFilter"
) -> bool:
    """
    Check if a Clang-CUDA version exist, which supports the CUDA SDK version in the row.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if only_cuda_compiler_backends(
        bashi.get_valid_compiler_backend_combinations(
            row, get_allowed_backend_combinations(), get_used_backends()
        )
    ):
        if (
            RT_CLANG_CUDA_MAX_CUDA_SUPPORT in alpaka_filter.runtime_infos
            and only_clang_cuda_compiler_backends(
                bashi.get_valid_compiler_backend_combinations(
                    row, get_allowed_backend_combinations(), get_used_backends()
                )
            )
            and ALPAKA_ACC_GPU_CUDA_ENABLE in row
            and not alpaka_filter.runtime_infos[RT_CLANG_CUDA_MAX_CUDA_SUPPORT](
                row[ALPAKA_ACC_GPU_CUDA_ENABLE].version
            )
        ):
            alpaka_filter.reason(
                "There is no Clang-CUDA version in the combination list, which supports the "
                f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} SDK."
            )
            return False
    return True


class AlpakaFilter(bashi.FilterBase):
    """Alpaka specific filter rules."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        version_relation: bashi.VersionRelation = bashi.VersionRelation(),
        output: IO[bashi.Parameter] | None = None,
    ):
        super().__init__(runtime_infos, version_relation, output)

    def __call__(
        self,
        row: bashi.BashiRow,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (bashi.BashiRow): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """
        # The order of the rule numbers (a1, a2, ...) corresponds to the order in which they are
        # implemented. However, it makes sense to call them in a different order in order to cancel
        # the filter as early as possible and thus improve performance.
        return (
            check_only_valid_backend_combinations_a1(row, self)
            and check_cuda_sdk_host_compiler_a2(row, self)
            and check_existing_clang_cuda_for_cuda_sdk_version_a6(row, self)
            and check_debug_build_hip_a3(row, self)
            and check_ubuntu_22_04_specifics_a4(row, self)
            and check_clang_16_and_older_a5(row, self)
        )
