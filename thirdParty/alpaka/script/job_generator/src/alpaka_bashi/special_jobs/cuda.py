"""Special Nvcc CUDA based CI Jobs"""

from typing import Dict, Any
from typeguard import typechecked

import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.ci_yaml.writer import construct_job_yaml
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=duplicate-code


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
@typechecked
def get_nvcc_relocatable_device_code_job(
    nvcc_version: str,
    gcc_version: str,
    cmake_version: str,
    container_version: str,
    stage_name: str,
    image_check: bool,
) -> Dict[str, Any]:
    """Add nvcc job, which enables the relocatable code function.

    Args:
        nvcc_version (str): Nvcc version
        gcc_version (str): Gcc version
        cmake_version (str): CMake version
        container_version (str): Container version
        stage_name (str): Stage name. If empty do not set an stage property.
        image_check (bool): Check if image exist. If not, use fallback image.

    Returns:
        Dict[str, Any]: GitLab CI yaml.
    """

    job_body = construct_job_yaml(
        combination=bashi.parse_combination(
            [
                (HOST_COMPILER, GCC, gcc_version),
                (DEVICE_COMPILER, NVCC, nvcc_version),
                (CMAKE, cmake_version),
                (UBUNTU, "24.04"),
                (CXX_STANDARD, "20"),
                (BUILD_TYPE, CMAKE_DEBUG),
                (MDSPAN, OFF),
                (JOB_EXECUTION_TYPE, JOB_EXECUTION_COMPILE_ONLY),
                (CI_PIPELINE_NAME, CI_PIPELINE_SPECIAL_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, nvcc_version),
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

    job_body["variables"]["alpaka_RELOCATABLE_DEVICE_CODE"] = "ON"
    job_body["variables"]["alpaka_CUDA_SHOW_REGISTER"] = "OFF"
    job_body["variables"]["alpaka_CUDA_KEEP_FILES"] = "OFF"
    job_body["variables"]["alpaka_CUDA_EXPT_EXTENDED_LAMBDA"] = "OFF"

    return {
        f"linux_special_nvcc{nvcc_version}_gcc{gcc_version}"
        "_debug_relocatable_device_code_compile_only": job_body
    }


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
@typechecked
def get_nvcc_extended_lambda_off_job(
    nvcc_version: str,
    gcc_version: str,
    cmake_version: str,
    container_version: str,
    stage_name: str,
    image_check: bool,
) -> Dict[str, Any]:
    """Add nvcc job, which disables the extended lambda support.

    Args:
        nvcc_version (str): Nvcc version
        gcc_version (str): Gcc version
        cmake_version (str): CMake version
        container_version (str): Container version
        stage_name (str): Stage name. If empty do not set an stage property.
        image_check (bool): Check if image exist. If not, use fallback image.

    Returns:
        Dict[str, Any]: GitLab CI yaml.
    """

    job_body = construct_job_yaml(
        combination=bashi.parse_combination(
            [
                (HOST_COMPILER, GCC, gcc_version),
                (DEVICE_COMPILER, NVCC, nvcc_version),
                (CMAKE, cmake_version),
                (UBUNTU, "24.04"),
                (CXX_STANDARD, "20"),
                (BUILD_TYPE, CMAKE_RELEASE),
                (MDSPAN, OFF),
                (JOB_EXECUTION_TYPE, JOB_EXECUTION_COMPILE_ONLY),
                (CI_PIPELINE_NAME, CI_PIPELINE_SPECIAL_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, nvcc_version),
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

    job_body["variables"]["alpaka_RELOCATABLE_DEVICE_CODE"] = "OFF"
    job_body["variables"]["alpaka_CUDA_SHOW_REGISTER"] = "OFF"
    job_body["variables"]["alpaka_CUDA_KEEP_FILES"] = "OFF"
    job_body["variables"]["alpaka_CUDA_EXPT_EXTENDED_LAMBDA"] = "OFF"

    return {
        f"linux_special_nvcc{nvcc_version}_gcc{gcc_version}"
        "_release_extended_lambda_off_compile_only": job_body
    }


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
@typechecked
def get_cuda_only_job(
    nvcc_version: str,
    gcc_version: str,
    cmake_version: str,
    container_version: str,
    stage_name: str,
    image_check: bool,
) -> Dict[str, Any]:
    """Add nvcc job, which uses the CUDA only mode.

    Args:
        nvcc_version (str): Nvcc version
        gcc_version (str): Gcc version
        cmake_version (str): CMake version
        container_version (str): Container version
        stage_name (str): Stage name. If empty do not set an stage property.
        image_check (bool): Check if image exist. If not, use fallback image.

    Returns:
        Dict[str, Any]: GitLab CI yaml.
    """

    job_body = construct_job_yaml(
        combination=bashi.parse_combination(
            [
                (HOST_COMPILER, GCC, gcc_version),
                (DEVICE_COMPILER, NVCC, nvcc_version),
                (CMAKE, cmake_version),
                (UBUNTU, "24.04"),
                (CXX_STANDARD, "20"),
                (BUILD_TYPE, CMAKE_RELEASE),
                (MDSPAN, OFF),
                (JOB_EXECUTION_TYPE, JOB_EXECUTION_COMPILE_ONLY),
                (CI_PIPELINE_NAME, CI_PIPELINE_SPECIAL_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, nvcc_version),
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

    job_body["variables"]["alpaka_ACC_GPU_CUDA_ONLY_MODE"] = "ON"
    job_body["variables"]["alpaka_ACC_GPU_HIP_ONLY_MODE"] = "OFF"
    job_body["variables"]["alpaka_RELOCATABLE_DEVICE_CODE"] = "OFF"
    job_body["variables"]["alpaka_CUDA_SHOW_REGISTER"] = "OFF"
    job_body["variables"]["alpaka_CUDA_KEEP_FILES"] = "OFF"
    job_body["variables"]["alpaka_CUDA_EXPT_EXTENDED_LAMBDA"] = "ON"

    return {
        f"linux_special_nvcc{nvcc_version}_gcc{gcc_version}"
        "_release_extended_lambda_off_compile_only": job_body
    }


# pylint: enable=duplicate-code
