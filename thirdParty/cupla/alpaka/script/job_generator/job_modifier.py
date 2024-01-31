"""Copyright 2023 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Contains alpaka specific function to manipulate the values of a generated job matrix.
"""

from typing import List, Dict, Tuple
from packaging import version

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import versions


def add_job_parameters(job_matrix: List[Dict[str, Tuple[str, str]]]):
    """Add additional job parameters, which are depend of other parameters and has nothing to do
    with combinations. For example, decide if a job is a compile only or runtime test job

    Args:
        job_matrix (List[Dict[str, Tuple[str, str]]]): Job matrix
    """
    for hip_version in versions.sw_versions[HIPCC]:
        for job in job_matrix:
            if (
                job[DEVICE_COMPILER][NAME] == HIPCC
                and job[DEVICE_COMPILER][VERSION] == hip_version
                and job[BUILD_TYPE][VERSION] == CMAKE_DEBUG
            ):
                job[JOB_EXECUTION_TYPE] = (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME)
                break

    # Disabled until all runtime errors for the SYCL CPU back-end are fixed.
    # for icpx_version in versions.sw_versions[ICPX]:
    #    for job in job_matrix:
    #        if(
    #            job[DEVICE_COMPILER][NAME] == ICPX
    #            and job[DEVICE_COMPILER][VERSION] == icpx_version
    #            and job[BUILD_TYPE][VERSION] == CMAKE_DEBUG
    #        ):
    #            job[JOB_EXECUTION_TYPE] = (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME)
    #            break

    ##############################################
    ## Defining runtime test for the CUDA backend
    ##############################################
    # The geneal rule is, that the latest minor release of each CUDA major release is tested with
    # all host compiler (one job per host compiler version)

    # This is a helper dictionary to find the latest minor version of a CUDA SDK major, used by a
    # specific host compiler.
    # e.g. We have CUDA SDK versions from 11.0 to 11.8 and 12.0 and 12.1. GCC as host compiler
    # supports all SDK versions, Clang as host compiler only the 11 versions and Clang as CUDA
    # compiler only up to 11.5. So the result is (see CUDA_SDK_per_compiler later):
    # {"GCC" : ["11.8", "12.1"], "Clang" : ["11.8"], "Clang-CUDA" : ["11.5"]}

    # { compiler_name : {major : (minor, (version_string)}}
    latest_CUDA_SDK_minor_versions: Dict[str : Dict[int:(int, str)]] = {
        GCC: {},
        CLANG: {},
        CLANG_CUDA: {},
    }

    MINOR_VERSION = 0
    VERSION_STRING = 1

    for job in job_matrix:
        if (
            ALPAKA_ACC_GPU_CUDA_ENABLE in job
            and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
        ):
            v = version.parse(job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION])
            if not v.major in latest_CUDA_SDK_minor_versions[job[HOST_COMPILER][NAME]]:
                latest_CUDA_SDK_minor_versions[job[HOST_COMPILER][NAME]][v.major] = (
                    v.minor,
                    job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION],
                )
            elif (
                latest_CUDA_SDK_minor_versions[job[HOST_COMPILER][NAME]][v.major][
                    MINOR_VERSION
                ]
                < v.minor
            ):
                latest_CUDA_SDK_minor_versions[job[HOST_COMPILER][NAME]][v.major] = (
                    v.minor,
                    job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION],
                )

    CUDA_SDK_per_compiler: Dict[str : List[str]] = {}

    for compiler_name, version_values in latest_CUDA_SDK_minor_versions.items():
        CUDA_SDK_per_compiler[compiler_name] = []
        for sdk in version_values.values():
            CUDA_SDK_per_compiler[compiler_name].append(sdk[VERSION_STRING])

    # this stores, if a job with a specific host compiler version is already tagged as runtime job
    used_host_compiler: Dict[str, List[str]] = {}

    for compiler_name in CUDA_SDK_per_compiler.keys():
        used_host_compiler[compiler_name] = []

    for job in job_matrix:
        for compiler_name, sdk_versions in CUDA_SDK_per_compiler.items():
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in job
                and job[HOST_COMPILER][NAME] == compiler_name
            ):
                for sdk_version in sdk_versions:
                    if (
                        job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] == sdk_version
                        # needs to be a release build, otherwise there is the risk of running ot of
                        # GPU resources
                        and job[BUILD_TYPE][VERSION] == CMAKE_RELEASE
                        and not job[HOST_COMPILER][VERSION]
                        in used_host_compiler[job[HOST_COMPILER][NAME]]
                    ):
                        used_host_compiler[job[HOST_COMPILER][NAME]].append(
                            job[HOST_COMPILER][VERSION]
                        )
                        job[JOB_EXECUTION_TYPE] = (
                            JOB_EXECUTION_TYPE,
                            JOB_EXECUTION_RUNTIME,
                        )

    # test one job per nvcc version with SM level of the CI GPU and highest supported SM level
    missing_nvcc_versions = versions.sw_versions[NVCC][:]

    STANDARD_SM_LEVEL = "61"
    for job in job_matrix:
        if (
            job[DEVICE_COMPILER][NAME] == NVCC
            and job[DEVICE_COMPILER][VERSION] in missing_nvcc_versions
            and job[JOB_EXECUTION_TYPE][VERSION] == JOB_EXECUTION_COMPILE_ONLY
        ):
            if version.parse(job[DEVICE_COMPILER][VERSION]) < version.parse("11.1"):
                job[SM_LEVEL] = (SM_LEVEL, STANDARD_SM_LEVEL + ";80")
            elif version.parse(job[DEVICE_COMPILER][VERSION]) < version.parse("11.5"):
                job[SM_LEVEL] = (SM_LEVEL, STANDARD_SM_LEVEL + ";86")
            elif version.parse(job[DEVICE_COMPILER][VERSION]) < version.parse("11.8"):
                job[SM_LEVEL] = (SM_LEVEL, STANDARD_SM_LEVEL + ";87")
            else:
                job[SM_LEVEL] = (SM_LEVEL, STANDARD_SM_LEVEL + ";90")
            missing_nvcc_versions.remove(job[DEVICE_COMPILER][VERSION])
        elif (
            ALPAKA_ACC_GPU_CUDA_ENABLE in job
            and ALPAKA_ACC_GPU_CUDA_ENABLE[VERSION] != OFF_VER
        ):
            job[SM_LEVEL] = (SM_LEVEL, STANDARD_SM_LEVEL)
        else:
            job[SM_LEVEL] = (SM_LEVEL, "")

    # run tests each time if a CPU backend is used
    for job in job_matrix:
        if job[DEVICE_COMPILER][NAME] in [GCC, CLANG]:
            job[JOB_EXECUTION_TYPE] = (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME)
