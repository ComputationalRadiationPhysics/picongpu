"""Generate special clang analysis job."""

from typing import Dict, Any
from typeguard import typechecked

import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.ci_yaml.writer import construct_job_yaml
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


# pylint: disable=duplicate-code
@typechecked
def get_clang_debug_analysis_job(
    clang_version: str,
    cmake_version: str,
    container_version: str,
    stage_name: str,
    image_check: bool,
) -> Dict[str, Any]:
    """Generate special Clang debug analysis job.

    Args:
        clang_version (str): Clang version
        cmake_version (str): CMake version
        container_version (str): Container version
        stage_name (str): Stage name. If empty do not set an stage property.
        image_check (bool): Check if image exist. If not, use fallback image.

    Returns:
        Dict[str, Any]: GitLab CI Yaml for Clang debug analysis job.
    """
    job_body = construct_job_yaml(
        combination=bashi.parse_combination(
            [
                (HOST_COMPILER, CLANG, clang_version),
                (DEVICE_COMPILER, CLANG, clang_version),
                (CMAKE, cmake_version),
                (UBUNTU, "24.04"),
                (CXX_STANDARD, 20),
                (BUILD_TYPE, CMAKE_DEBUG),
                (MDSPAN, OFF),
                (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME),
                (CI_PIPELINE_NAME, CI_PIPELINE_SPECIAL_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_CPU_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_GPU_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF),
            ]
        ),
        stage=stage_name,
        container_version=container_version,
        image_check=image_check,
    )

    job_body["variables"]["ALPAKA_CI_ANALYSIS"] = "ON"
    job_body["variables"]["ALPAKA_CI_STDLIB"] = "libc++"
    job_body["variables"]["CMAKE_INSTALL_PREFIX"] = "${CI_PROJECT_DIR}/_install"
    job_body["variables"]["ALPAKA_CI_ANALYSIS"] = "ON"
    job_body["variables"]["alpaka_DEBUG"] = 2

    return {f"linux_special_clang-{clang_version}_debug_analysis": job_body}


@typechecked
def get_clang_asan_job(
    clang_version: str,
    cmake_version: str,
    container_version: str,
    stage_name: str,
    image_check: bool,
) -> Dict[str, Any]:
    """Generate special Clang asan sanitizer job.

    Args:
        clang_version (str): Clang version
        cmake_version (str): CMake version
        container_version (str): Container version
        stage_name (str): Stage name. If empty do not set an stage property.
        image_check (bool): Check if image exist. If not, use fallback image.

    Returns:
        Dict[str, Any]: GitLab CI Yaml for Clang asan sanitizer job.
    """
    cpp_standard = "20"

    job_body = construct_job_yaml(
        combination=bashi.parse_combination(
            [
                (HOST_COMPILER, CLANG, clang_version),
                (DEVICE_COMPILER, CLANG, clang_version),
                (CMAKE, cmake_version),
                (UBUNTU, "24.04"),
                (CXX_STANDARD, cpp_standard),
                (BUILD_TYPE, CMAKE_RELEASE_WITH_DEBUG_INFO),
                (MDSPAN, OFF),
                (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME),
                (CI_PIPELINE_NAME, CI_PIPELINE_SPECIAL_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_CPU_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_GPU_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF),
            ]
        ),
        stage=stage_name,
        container_version=container_version,
        image_check=image_check,
    )

    job_body["variables"]["ALPAKA_CI_STDLIB"] = "libstdc++"
    job_body["variables"]["CMAKE_CXX_EXTENSIONS"] = "OFF"
    job_body["variables"]["ALPAKA_CI_SANITIZERS"] = "ASan"

    return {f"linux_special_clang-{clang_version}_relwithdebinfo_asan_c++{cpp_standard}": job_body}


# pylint: enable=duplicate-code
