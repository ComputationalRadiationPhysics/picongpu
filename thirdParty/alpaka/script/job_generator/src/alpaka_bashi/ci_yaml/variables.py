"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the variables of the GitLab CI test job yaml.
"""

from typing import Dict, Any
from typeguard import typechecked
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.globals
from alpaka_bashi.globals import (
    ALPAKA_ACC_SYCL_ENABLE,
    BUILD_TYPE,
    JOB_EXECUTION_TYPE,
    JOB_EXECUTION_RUNTIME_VER,
    CI_PIPELINE_COMPILE_ONLY_VER,
    MDSPAN,
)
from alpaka_bashi.versions import get_used_backends


@typechecked
def set_generic_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which all jobs requires.

    Args:
        job_body (Dict[str, Any]): alpaka ci job
        combination (bashi.Combination): combination
    """
    variables = job_body["variables"]
    variables |= {
        "alpaka_CI": "GITLAB",
        "ALPAKA_CI_OS_NAME": "Linux",
        "alpaka_DEBUG": 0,
        "alpaka_ENABLE_WERROR": "ON",
        "ALPAKA_CI_ANALYSIS": "OFF",
        "ALPAKA_CI_SANITIZERS": "",
        "ALPAKA_CI_BUILD_JOBS": "$CI_CPUS",
        "OMP_NUM_THREADS": "$CI_CPUS",
        "alpaka_ACC_GPU_CUDA_ONLY_MODE": "OFF",
        "alpaka_ACC_GPU_HIP_ONLY_MODE": "OFF",
        "ALPAKA_CI_STDLIB": "libstdc++",
    }

    variables["ALPAKA_CI_UBUNTU_VER"] = bashi.ubuntu_version_to_string(combination[UBUNTU].version)
    variables["CMAKE_BUILD_TYPE"] = alpaka_bashi.globals.get_version_aliases()[
        combination[BUILD_TYPE].name
    ][combination[BUILD_TYPE].version]

    variables["ALPAKA_CI_RUN_TESTS"] = (
        "OFF" if combination[JOB_EXECUTION_TYPE].version == CI_PIPELINE_COMPILE_ONLY_VER else "ON"
    )
    variables["ALPAKA_CI_CMAKE_VER"] = str(combination[CMAKE].version)
    variables["alpaka_CXX_STANDARD"] = str(combination[CXX_STANDARD].version)
    variables["ALPAKA_TEST_MDSPAN"] = bashi.on_off_ver_to_str(combination[MDSPAN].version)


@typechecked
def set_backend_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set all backend variables. A backend is ether enabled or disabled. Backend specific
    variables will be set another functions."""
    for backend in get_used_backends():
        if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
            job_body["variables"][ALPAKA_ACC_GPU_CUDA_ENABLE] = (
                "OFF" if combination[backend].version == OFF_VER else "ON"
            )
        elif backend in bashi.ONE_API_BACKENDS:

            def one_api_backend_is_enabled():
                for one_api_backend in bashi.ONE_API_BACKENDS:
                    if combination[one_api_backend].version == ON_VER:
                        return True
                return False

            job_body["variables"][ALPAKA_ACC_SYCL_ENABLE] = (
                "ON" if one_api_backend_is_enabled() else "OFF"
            )
        else:
            job_body["variables"][backend] = bashi.on_off_ver_to_str(combination[backend].version)


@typechecked
def set_gcc_device_compiler_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which are specific if the gcc is the device compiler."""
    job_body["variables"]["ALPAKA_CI_CXX"] = "g++"
    job_body["variables"]["ALPAKA_CI_GCC_VER"] = str(combination[DEVICE_COMPILER].version)
    if combination[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE].version == ON_VER:
        job_body["variables"]["ALPAKA_CI_TBB_VERSION"] = "2021.10.0"


@typechecked
def set_clang_device_compiler_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which are specific if the clang is the device compiler."""
    job_body["variables"]["ALPAKA_CI_CXX"] = "clang++"
    job_body["variables"]["ALPAKA_CI_CLANG_VER"] = str(combination[DEVICE_COMPILER].version)
    if combination[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE].version == ON_VER:
        job_body["variables"]["ALPAKA_CI_TBB_VERSION"] = "2021.10.0"


@typechecked
def set_hipcc_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which are specific if the hipcc is the device compiler."""

    def get_clang_version(hipcc_version):
        for hipcc_clang in bashi.version.dependencies.hipcc.HIPCC_CLANG_VERSION:
            if hipcc_clang.compiler == hipcc_version:
                return hipcc_clang.clang
        raise RuntimeError(f"No Clang version for hipcc {hipcc_version}")

    job_body["variables"]["ALPAKA_CI_HIP_VERSION"] = str(combination[DEVICE_COMPILER].version)
    job_body["variables"]["CMAKE_HIP_COMPILER"] = "clang++"
    job_body["variables"]["ALPAKA_CI_CXX"] = "clang++"
    job_body["variables"]["ALPAKA_CI_CLANG_VER"] = str(
        get_clang_version(combination[DEVICE_COMPILER].version)
    )
    job_body["variables"]["CMAKE_HIP_ARCHITECTURES"] = "${CI_GPU_ARCH}"
    job_body["variables"]["ALPAKA_CI_STDLIB"] = "libstdc++"


@typechecked
def get_sm_level(combination: bashi.Combination) -> str:
    """Get the CUDA SM level depending on the combination"""
    # SM Level of the Nvidia Quadro P5000
    # Compile also for the SM level even it is not used on the Quadro P500
    nvidia_gpu_ci_runner_sm_level = "80"

    if combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version < packaging.version.parse("13.0"):
        # SM Level of the Nvidia Quadro P5000
        # with CUDA 13.0 SM 6.1 is deprecated
        nvidia_gpu_ci_runner_sm_level += ";61"

    if combination[JOB_EXECUTION_TYPE].version == JOB_EXECUTION_RUNTIME_VER:
        return nvidia_gpu_ci_runner_sm_level

    if combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= packaging.version.parse("12.0"):
        if combination[DEVICE_COMPILER].name == NVCC:
            return nvidia_gpu_ci_runner_sm_level + ";90"

        # Clang-CUDA does not support SM 9.0 with CUDA 12.0, 12.1 is the minimum
        if combination[DEVICE_COMPILER].name == CLANG_CUDA and combination[
            ALPAKA_ACC_GPU_CUDA_ENABLE
        ].version >= packaging.version.parse("12.1"):
            return nvidia_gpu_ci_runner_sm_level + ";90"

    return nvidia_gpu_ci_runner_sm_level


@typechecked
def set_nvcc_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which are specific if the nvcc is the device compiler."""

    job_body["variables"]["ALPAKA_CI_CUDA_COMPILER"] = "nvcc"
    job_body["variables"]["ALPAKA_CI_STDLIB"] = "libstdc++"
    job_body["variables"]["CMAKE_CUDA_ARCHITECTURES"] = get_sm_level(combination)
    job_body["variables"]["ALPAKA_CI_CUDA_VERSION"] = str(
        combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version
    )
    job_body["variables"]["alpaka_RELOCATABLE_DEVICE_CODE"] = "OFF"
    job_body["variables"]["alpaka_CUDA_SHOW_REGISTER"] = "OFF"
    job_body["variables"]["alpaka_CUDA_KEEP_FILES"] = "OFF"
    # mdspan requires experimental extended lambda
    job_body["variables"]["alpaka_CUDA_EXPT_EXTENDED_LAMBDA"] = bashi.on_off_ver_to_str(
        combination[MDSPAN].version
    )
    if combination[HOST_COMPILER].name == GCC:
        job_body["variables"]["ALPAKA_CI_CXX"] = "g++"
        job_body["variables"]["ALPAKA_CI_GCC_VER"] = str(combination[HOST_COMPILER].version)
    elif combination[HOST_COMPILER].name == CLANG:
        job_body["variables"]["ALPAKA_CI_CXX"] = "clang++"
        job_body["variables"]["ALPAKA_CI_CLANG_VER"] = str(combination[HOST_COMPILER].version)
    else:
        raise RuntimeError(f"Unknown nvcc host compiler {combination[HOST_COMPILER].name}")


@typechecked
def set_clang_cuda_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which are specific if the clang-cuda is the device compiler."""

    job_body["variables"]["ALPAKA_CI_CUDA_COMPILER"] = "clang++"
    job_body["variables"]["ALPAKA_CI_CXX"] = "clang++"
    job_body["variables"]["ALPAKA_CI_CLANG_VER"] = str(combination[DEVICE_COMPILER].version)
    job_body["variables"]["ALPAKA_CI_STDLIB"] = "libstdc++"
    job_body["variables"]["CMAKE_CUDA_ARCHITECTURES"] = get_sm_level(combination)
    job_body["variables"]["ALPAKA_CI_CUDA_VERSION"] = str(
        combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version
    )
    job_body["variables"]["alpaka_RELOCATABLE_DEVICE_CODE"] = "OFF"
    job_body["variables"]["alpaka_CUDA_SHOW_REGISTER"] = "OFF"
    job_body["variables"]["alpaka_CUDA_KEEP_FILES"] = "OFF"
    job_body["variables"]["alpaka_CUDA_EXPT_EXTENDED_LAMBDA"] = "OFF"


@typechecked
def set_icpx_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set variables which are specific if the icpx is the device compiler."""

    def get_clang_version(icpx_version):
        for icpx_clang in bashi.version.dependencies.icpx.ICPX_CLANG_VERSION:
            if icpx_clang.compiler == icpx_version:
                return icpx_clang.clang
        raise RuntimeError(f"No Clang version for icpx {icpx_version}")

    backend_to_cmake_arg = {
        ALPAKA_ACC_ONEAPI_CPU_ENABLE: "alpaka_SYCL_ONEAPI_CPU",
        ALPAKA_ACC_ONEAPI_GPU_ENABLE: "alpaka_SYCL_ONEAPI_GPU",
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE: "alpaka_SYCL_ONEAPI_FPGA",
    }

    job_body["variables"]["ALPAKA_CI_CXX"] = "icpx"
    job_body["variables"]["ALPAKA_CI_CLANG_VER"] = str(
        get_clang_version(combination[DEVICE_COMPILER].version)
    )
    job_body["variables"]["ALPAKA_CI_ONEAPI_VERSION"] = str(combination[DEVICE_COMPILER].version)
    job_body["variables"]["ALPAKA_CI_STDLIB"] = "libstdc++"
    for backend, cmake_arg in backend_to_cmake_arg.items():
        job_body["variables"][cmake_arg] = bashi.on_off_ver_to_str(combination[backend].version)

    if combination[ALPAKA_ACC_ONEAPI_CPU_ENABLE].version == ON_VER:
        job_body["variables"]["alpaka_SYCL_ONEAPI_CPU_ISA"] = "avx2"

    if combination[ALPAKA_ACC_ONEAPI_GPU_ENABLE].version == ON_VER:
        job_body["variables"]["alpaka_SYCL_ONEAPI_GPU_DEVICES"] = "spir64"

    if combination[ALPAKA_ACC_ONEAPI_FPGA_ENABLE].version == ON_VER:
        job_body["variables"]["alpaka_SYCL_ONEAPI_FPGA_MODE"] = "emulation"
        job_body["variables"]["alpaka_SYCL_ONEAPI_FPGA_BOARD"] = ""
        job_body["variables"]["alpaka_SYCL_ONEAPI_FPGA_BSP"] = ""


@typechecked
def set_variables(job_body: Dict[str, Any], combination: bashi.Combination):
    """Set the variables of the GitLab CI test job yaml depending on the combination.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    if "variables" not in job_body:
        job_body["variables"] = {}

    set_generic_variables(job_body, combination)
    set_backend_variables(job_body, combination)

    set_device_compiler_variables = {
        GCC: set_gcc_device_compiler_variables,
        CLANG: set_clang_device_compiler_variables,
        HIPCC: set_hipcc_variables,
        NVCC: set_nvcc_variables,
        CLANG_CUDA: set_clang_cuda_variables,
        ICPX: set_icpx_variables,
    }

    set_device_compiler_variables[combination[DEVICE_COMPILER].name](job_body, combination)
