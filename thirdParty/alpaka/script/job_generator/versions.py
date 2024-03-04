"""Copyright 2023 Simeon Ehrig, Jan Stephan, René Widera
SPDX-License-Identifier: MPL-2.0

Used software in the CI tests."""

from typing import Dict, List, Tuple
from typeguard import typechecked

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


sw_versions: Dict[str, List[str]] = {
    GCC: ["9", "10", "11", "12", "13"],
    CLANG: ["9", "10", "11", "12", "13", "14", "15", "16", "17"],
    NVCC: [
        "11.0",
        "11.1",
        "11.2",
        "11.3",
        "11.4",
        "11.5",
        "11.6",
        "11.7",
        "11.8",
        "12.0",
        "12.1",
        "12.2",
        "12.3",
    ],
    HIPCC: ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "6.0"],
    ICPX: ["2023.1.0", "2023.2.0"],
    # Contains all enabled back-ends.
    # There are special cases for ALPAKA_ACC_GPU_CUDA_ENABLE and ALPAKA_ACC_GPU_HIP_ENABLE
    # which have to be combined with nvcc and hipcc versions.
    BACKENDS: [
        # gcc and clang
        [
            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
            ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
        ],
        # ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        # nvcc
        [
            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
            ALPAKA_ACC_GPU_CUDA_ENABLE,
        ],
        # clang-cuda
        # OpenMP is not supported for clang as cuda compiler
        # https://github.com/alpaka-group/alpaka/issues/639
        # therefore a extra combination is required
        [
            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
            ALPAKA_ACC_GPU_CUDA_ENABLE,
        ],
        # hip
        [
            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            ALPAKA_ACC_GPU_HIP_ENABLE,
        ],
        # sycl (oneAPI)
        # Turn off OpenMP back-ends until Intel fixes https://github.com/intel/llvm/issues/10711
        [
            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
            ALPAKA_ACC_SYCL_ENABLE,
        ],
    ],
    UBUNTU: ["20.04"],
    CMAKE: ["3.22.6", "3.23.5", "3.24.4", "3.25.3", "3.26.4"],
    BOOST: [
        "1.74.0",
        "1.75.0",
        "1.76.0",
        "1.77.0",
        "1.78.0",
        "1.79.0",
        "1.80.0",
        "1.81.0",
        "1.82.0",
    ],
    CXX_STANDARD: ["17", "20"],
    BUILD_TYPE: BUILD_TYPES,
    # use only TEST_COMPILE_ONLY, because TEST_RUNTIME will be set manually depend on some
    # conditions later
    JOB_EXECUTION_TYPE: [JOB_EXECUTION_COMPILE_ONLY],
    MDSPAN: [ON_VER, OFF_VER],
}


@typechecked
def get_compiler_versions(clang_cuda: bool = True) -> List[Tuple[str, str]]:
    """Generate a list of compiler name version tuple.

    Args:
        clang_cuda (bool, optional): If true, create entries for cling-cuda basing on the clang
        version. Defaults to True.

    Returns:
        List[Tuple[str, str]]: The compiler name version tuple list.
    """
    compilers: List[Tuple[str, str]] = []

    # only use keys defined in sw_versions
    for compiler_name in set(sw_versions.keys()).intersection(
        [GCC, CLANG, NVCC, HIPCC, ICPX]
    ):
        for version in sw_versions[compiler_name]:
            compilers.append((compiler_name, version))
            if clang_cuda and compiler_name == CLANG:
                compilers.append((CLANG_CUDA, version))

    return compilers


@typechecked
def get_backend_matrix() -> List[List[Tuple[str, str]]]:
    """Generates a back-end list which contains different combinations of enabled back-ends.

    Returns:
        List[List[Tuple[str, str]]]: The backend list.
    """
    combination_matrix: List[List[Tuple[str, str]]] = []

    for backend_list in sw_versions[BACKENDS]:
        combination: List[Tuple[str, str]] = []
        # all back-ends except ALPAKA_ACC_GPU_HIP_ENABLE and
        # ALPAKA_ACC_GPU_CUDA_ENABLE can be enabled without having a version number attached to them
        for backend_name in backend_list:
            if backend_name not in [
                ALPAKA_ACC_GPU_HIP_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
            ]:
                combination.append((backend_name, ON_VER))

        # add a back-end entry for each CUDA SDK version
        if ALPAKA_ACC_GPU_CUDA_ENABLE in backend_list:
            for cuda_version in sw_versions[NVCC]:
                combination_copy = combination[:]
                combination_copy.append((ALPAKA_ACC_GPU_CUDA_ENABLE, cuda_version))
                combination_matrix.append(combination_copy)
        # add a back-end entry for each HIP SDK version
        elif ALPAKA_ACC_GPU_HIP_ENABLE in backend_list:
            for rocm_version in sw_versions[HIPCC]:
                combination_copy = combination[:]
                combination_copy.append((ALPAKA_ACC_GPU_HIP_ENABLE, rocm_version))
                combination_matrix.append(combination_copy)
        else:
            combination_matrix.append(combination)

    return combination_matrix


@typechecked
def get_sw_tuple_list(name: str) -> List[Tuple[str, str]]:
    """Creates a list of software name version tuples for a software name.

    Args:
        name (str): Name of the software

    Returns:
        List[Tuple[str, str]]: List of software name versions tuples.
    """
    tuple_list: List[Tuple[str, str]] = []
    for version in sw_versions[name]:
        tuple_list.append((name, version))

    return tuple_list
