"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Software versions to be tested.
"""

from copy import deepcopy
from typing import Dict, List, Union
import packaging.version
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version.dependencies.clang_cuda import CLANG_CUDA_MAX_CUDA_VERSION, ClangCudaSDKSupport
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

ALPAKA_VERSIONS: Dict[str, List[Union[str, int, float]]] = {
    GCC: [11, 12, 13],
    CLANG: [14, 15, 16, 17, 18, 19, 20],
    NVCC: [12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.8, 12.9, 13.0],
    HIPCC: [6.0, 6.1, 6.2, 6.3, 6.4, 7.0, 7.1, 7.2],
    ICPX: ["2025.0"],
    UBUNTU: ["22.04", "24.04"],
    CMAKE: ["3.25.3", "3.26.4", "3.27.9", "3.28.6", "3.29.8", "3.30.3"],
    CXX_STANDARD: ["20"],
    BUILD_TYPE: BUILD_TYPES,
    MDSPAN: [ON, OFF],
}


def _get_clang_cuda_versions() -> List[Union[str, int, float]]:
    """Return a list of Clang-CUDA versions. If there is no CUDA version
    bashi.versions.CLANG_CUDA_MAX_CUDA_VERSION which supports a specific Clang-CUDA, don't it add to
    the list.

    Returns:
        List[Union[str, int, float]]: List of Clang-CUDA versions.
    """
    min_cuda_version = packaging.version.parse(str(min(ALPAKA_VERSIONS[NVCC])))
    min_clang_cuda_version = packaging.version.parse("0")
    for clang_cuda_sdk in sorted(get_alpaka_version_relation().get_clang_cuda_max_cuda_version()):
        if min_cuda_version <= clang_cuda_sdk.cuda:
            min_clang_cuda_version = clang_cuda_sdk.clang_cuda
            break
    return [
        ver
        for ver in ALPAKA_VERSIONS[CLANG]
        if packaging.version.parse(str(ver)) >= min_clang_cuda_version
    ]


def get_used_compiler_versions() -> dict[str, list[str | int | float]]:
    """Return a dict of used compiler and it's versions.

    Returns:
        dict[str, list[str | int | float]]: The key is the compiler name and value contains all
        versions.
    """
    return {name: versions for name, versions in ALPAKA_VERSIONS.items() if name in COMPILERS} | {
        CLANG_CUDA: _get_clang_cuda_versions()
    }


def get_used_backends() -> list[str]:
    """Return the list of backends, used by alpaka."""
    return [
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
        ALPAKA_ACC_ONEAPI_CPU_ENABLE,
        ALPAKA_ACC_ONEAPI_GPU_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
    ]


def get_allowed_backend_combinations() -> list[bashi.CompilerBackendCombination]:
    """Return list of enabled backends for different host and device compiler combinations."""
    allowed_nvcc_backends: List[ValueName] = [
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
    ]

    # Turn off OpenMP back-ends until Intel fixes https://github.com/intel/llvm/issues/10711
    allowed_icpx_backends: List[ValueName] = [
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ]

    return [
        bashi.CompilerBackendCombination(
            GCC,
            GCC,
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
            ],
        ),
        bashi.CompilerBackendCombination(
            CLANG,
            CLANG,
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
            ],
        ),
        bashi.CompilerBackendCombination(
            GCC,
            NVCC,
            allowed_nvcc_backends,
        ),
        bashi.CompilerBackendCombination(
            CLANG,
            NVCC,
            allowed_nvcc_backends,
        ),
        # OpenMP is not supported for clang as cuda compiler
        # https://github.com/alpaka-group/alpaka/issues/639
        bashi.CompilerBackendCombination(
            CLANG_CUDA,
            CLANG_CUDA,
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            ],
        ),
        bashi.CompilerBackendCombination(
            HIPCC,
            HIPCC,
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
        bashi.CompilerBackendCombination(
            ICPX,
            ICPX,
            allowed_icpx_backends
            + [
                ALPAKA_ACC_ONEAPI_CPU_ENABLE,
            ],
        ),
        bashi.CompilerBackendCombination(
            ICPX,
            ICPX,
            allowed_icpx_backends
            + [
                ALPAKA_ACC_ONEAPI_GPU_ENABLE,
            ],
        ),
        bashi.CompilerBackendCombination(
            ICPX,
            ICPX,
            allowed_icpx_backends
            + [
                ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
            ],
        ),
    ]


def get_software_versions_for_alpaka() -> Dict[str, List[Union[str, int, float]]]:
    """Return dict of all compiler and software versions, which should be used as input for the
    combination generator.

    Raises:
        RuntimeError: If no valid Clang-CUDA versions exist.

    Returns:
        Dict[str, List[Union[str, int, float]]]: List of compiler and software versions.
    """

    clang_cuda_versions = _get_clang_cuda_versions()
    # The alpaka filter function cannot handle the case, that Clang-CUDA compiler are missing.
    # In the case, that the parameter-value-matrix is missing Clang-CUDA, we get a meaning full
    # error.
    if len(clang_cuda_versions) == 0:
        raise RuntimeError("Alpaka custom filter does not work without Clang-CUDA version.")

    return deepcopy(ALPAKA_VERSIONS) | {CLANG_CUDA: clang_cuda_versions}


def get_alpaka_version_relation() -> bashi.VersionRelation:
    """Returns:
    bashi.VersionRelation: bashi.VersionRelation object with alpaka specific modifications.
    """
    # bashi already offers numerous software relations. You can find all predefined relations
    # here: https://github.com/alpaka-group/bashi/blob/main/src/bashi/version/relation.py
    #
    # Relationships can be easily extended. The following example assumes that bashi has already
    # defined the relationship for Clang-CUDA 7 up to 17 and the CUDA SDK. The relationship is to be
    # extended up to Clang-CUDA 22.
    #
    # clang_cuda_max_cuda_version = CLANG_CUDA_MAX_CUDA_VERSION + [
    #    ClangCudaSDKSupport("18", "12.3"),
    #    ClangCudaSDKSupport("22", "13.0"),
    # ]
    #
    # bashi.VersionRelation(clang_cuda_max_cuda_version=clang_cuda_max_cuda_version)

    clang_cuda_max_cuda_version = CLANG_CUDA_MAX_CUDA_VERSION + [
        ClangCudaSDKSupport("18", "12.3"),
        ClangCudaSDKSupport("22", "13.0"),
    ]

    return bashi.VersionRelation(clang_cuda_max_cuda_version=clang_cuda_max_cuda_version)
