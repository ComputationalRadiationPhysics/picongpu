"""Used software in the CI tests."""

from typing import Dict, List, Tuple
from typeguard import typechecked

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


# TODO: only an example
# sw_versions: Dict[str, List[str]] = {
#     GCC: ["7", "8", "9", "10", "11"],
#     CLANG: ["7", "8", "9", "10", "11", "12", "13", "14", "15"],
#     NVCC: ["11.0", "11.1", "11.2", "11.3", "11.4", "11.5", "11.6"],
#     HIPCC: ["4.3", "4.5", "5.0", "5.1", "5.2"],
#     BACKENDS: [
#         ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
#         ALPAKA_ACC_GPU_CUDA_ENABLE,
#         # ALPAKA_ACC_GPU_HIP_ENABLE,
#     ],
#     UBUNTU: ["20.04"],
#     CMAKE: ["3.18.6", "3.19.8", "3.20.6", "3.21.6", "3.22.3"],
#     BOOST: [
#         "1.74.0",
#         "1.75.0",
#         "1.76.0",
#         "1.77.0",
#         "1.78.0",
#     ],
#     CXX_STANDARD: ["17", "20"],
# }

sw_versions: Dict[str, List[str]] = {
    GCC: [],
    CLANG: [],
    NVCC: [],
    HIPCC: [],
    BACKENDS: [],
    UBUNTU: [],
    CMAKE: [],
    BOOST: [],
    CXX_STANDARD: [],
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
        [GCC, CLANG, NVCC, HIPCC]
    ):
        for version in sw_versions[compiler_name]:
            compilers.append((compiler_name, version))
            if clang_cuda and compiler_name == CLANG:
                compilers.append((CLANG_CUDA, version))

    return compilers


@typechecked
def get_backend_matrix() -> List[List[Tuple[str, str]]]:
    """Generate backend list, where only backend is active on the same time.

    Returns:
        List[List[Tuple[str, str]]]: The backend list.
    """

    return []


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
