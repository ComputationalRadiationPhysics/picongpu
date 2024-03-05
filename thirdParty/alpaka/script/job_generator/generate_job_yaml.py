"""Copyright 2023 Simeon Ehrig, Jan Stephan
SPDX-License-Identifier: MPL-2.0

Create GitLab-CI job description written in yaml from the job matrix."""

from typing import List, Dict, Tuple
from typeguard import typechecked
import os, yaml
import gitlab


from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from util import print_warn, exit_error

JOB_COMPILE_ONLY = "compile_only_job"
JOB_RUNTIME = "runtime_job_gpu"
JOB_CPU_RUNTIME = "runtime_job_cpu"
JOB_ROCM_RUNTIME = "rocm_runtime_job"
JOB_NVCC_GCC_RUNTIME = "nvcc_gcc_runtime_job"
JOB_NVCC_CLANG_RUNTIME = "nvcc_clng_runtime_job"
JOB_CLANG_CUDA_RUNTIME = "clang_cuda_runtime_job"
JOB_ICPX_RUNTIME = "icpx_runtime_job"
JOB_UNKNOWN = "unknowm_job_type"

WAVE_GROUP_NAMES = [
    JOB_COMPILE_ONLY,
    JOB_RUNTIME,
    JOB_CPU_RUNTIME,
    # can be enabled again, if fine granular scheduling is required
    # JOB_ROCM_RUNTIME,
    # JOB_NVCC_GCC_RUNTIME,
    # JOB_NVCC_CLANG_RUNTIME,
    # JOB_CLANG_CUDA_RUNTIME,
    # JOB_ICPX_RUNTIME,
    JOB_UNKNOWN,
]

# is used to display missing image warning one time
image_warning_cache = []


@typechecked
def get_env_var_name(variable_name: str) -> str:
    """Transform string to a shape, which is allowed as environment variable.

    Args:
        variable_name (str): Variable name.

    Returns:
        str: Transformed variable name.
    """
    return variable_name.upper().replace("-", "_")


@typechecked
def job_prefix_coding(job: Dict[str, Tuple[str, str]]) -> str:
    """Generate prefix for job name, depending of the available software versions.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.

    Returns:
        str: Job name Prefix.
    """
    version_str = ""
    for sw in [CMAKE, BOOST, UBUNTU, CXX_STANDARD, MDSPAN, BUILD_TYPE]:
        if sw in job:
            if job[sw][NAME] == CXX_STANDARD:
                version_str += "_cxx" + job[sw][VERSION]
            elif job[sw][NAME] == BUILD_TYPE:
                version_str += "_" + job[sw][VERSION]
            elif job[sw][NAME] == MDSPAN:
                if job[sw][VERSION] == ON_VER:
                    version_str += "_mdspan"
            else:
                version_str += "_" + job[sw][NAME] + job[sw][VERSION]

    return version_str


@typechecked
def get_images_from_registry(container_version: str) -> List[str]:
    """Returns a list of all container images for a given tag.

    Args:
        container_version (str): Container tag

    Returns:
        List[str]: list of container images
    """
    # hard coding the url and project ID is fine, because there is only one
    # container registry
    gl = gitlab.Gitlab(url="https://codebase.helmholtz.cloud")
    project = gl.projects.get("4742")
    repositories = project.repositories.list(get_all=True)

    # uniform data structure
    if not isinstance(repositories, list):
        repositories = [repositories]

    images = []
    # this process is actual pretty slow
    # maybe it does a HTML request each time if we iterate over repo.tags.list()
    # and get a new tag
    for repo in repositories:
        for tag in repo.tags.list():
            if tag.attributes["name"] == container_version:
                images.append(tag.attributes["location"])

    return images


@typechecked
def job_image(
    job: Dict[str, Tuple[str, str]],
    container_version: float,
    gitlab_images: List[str] = {},
) -> str:
    """Generates the image URL depending on the host and device compiler and the
    job's selected back-end.

    If gitlab_images is not empty only existing images are returned.
    Otherwise, the name of the image is constructed by a set of rules. However,
    the resulting image may not exist. In this case an image is returned that
    best matches the software's requirements.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.
        container_version (float): Container version tag.
        gitlab_images (List[str]): List of existing container images. If the
            list is empty, the function does not check if an image exist.

    Returns:
        str: Full container url, which can be used with docker pull.
    """

    verified_container_url = [
        "registry.hzdr.de/crp/alpaka-group-container/"
        + "alpaka-ci-ubuntu"
        + job[UBUNTU][VERSION]
    ]

    is_in_gitlab_images = lambda name: bool(
        [i for i in gitlab_images if i.startswith(name)]
    )

    def verify_image(
        test_url: List[str], verified_url: List[str], gitlab_images: List[str]
    ) -> bool:
        """Verify if the test_url is included in gitlab_images.

        Args:
            test_url (List[str]): Image URL to test. The list will be
                concatenated with all whitespaces removed.
            verified_url (List[str]): Working URL. Only for the warning message.
                The Python list will be concatenated to a single string (URL)
                without whitespaces or chars between the list elements.
            gitlab_images (List[str]): List of existing container images.

        Returns:
            bool: Return True if images exist.
        """
        global image_warning_cache
        if gitlab_images and not is_in_gitlab_images("".join(test_url)):
            if "".join(test_url) not in image_warning_cache:
                print_warn(
                    f'image {"".join(test_url)} does not exist\n'
                    f'  use instead image: {"".join(verified_url)}'
                )
                # append image to a cache to show the warning only one time
                image_warning_cache.append("".join(test_url))
            return False
        else:
            return True

    # if the base image does not exist we have no fall back
    if gitlab_images and not is_in_gitlab_images("".join(verified_container_url)):
        exit_error(f'base image {"".join(verified_container_url)} does not exist')

    testing_container_url = verified_container_url.copy()

    # If only the GCC is used, use special gcc version of the container.
    if job[HOST_COMPILER][NAME] == GCC:
        testing_container_url.append("-gcc")

    if not verify_image(testing_container_url, verified_container_url, gitlab_images):
        return "".join(verified_container_url)
    verified_container_url = testing_container_url.copy()

    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        # Cast cuda version shape. E.g. from 11.0 to 110
        testing_container_url.insert(
            1, "-cuda" + str(int(float(job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]) * 10))
        )

    if not verify_image(testing_container_url, verified_container_url, gitlab_images):
        return "".join(verified_container_url)
    verified_container_url = testing_container_url.copy()

    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF_VER
    ):
        testing_container_url.insert(
            1, "-rocm" + job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION]
        )

    if not verify_image(testing_container_url, verified_container_url, gitlab_images):
        return "".join(verified_container_url)
    verified_container_url = testing_container_url.copy()

    # append container tag
    verified_container_url.append(":" + str(container_version))

    return "".join(verified_container_url)


@typechecked
def append_backend_variables(
    variables: Dict[str, str], job: Dict[str, Tuple[str, str]]
):
    """Searches for enabled back-ends in the job parameters and appends the back-end
    variable to variables to enable it in the CI job.

    Args:
        variables (Dict[str, str]): variables of the CI job
        job (Dict[str, Tuple[str, str]]): job with parameters from the pair-wise
            generator.
    """
    for backend in BACKENDS_LIST:
        if backend in job:
            variables[backend] = "ON"


@typechecked
def job_variables(job: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Add variables to the job depending of the job dict.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict

    Returns:
        Dict[str, str]: Dict of {variable name : variable value}.
    """
    variables: Dict[str, str] = {}

    ################################################################################################
    ### job independent environment variables
    ################################################################################################

    # OS name
    # at the moment, the GitLab CI has only Linux Runner, therefore it is fine to hard code it
    variables["ALPAKA_CI_OS_NAME"] = "Linux"
    variables["ALPAKA_CI_BUILD_JOBS"] = "$CI_CPUS"
    variables["OMP_NUM_THREADS"] = "$CI_CPUS"
    # the variable is required, that the test scripts, on which CI they are running
    variables["alpaka_CI"] = "GITLAB"
    variables["ALPAKA_CI_ANALYSIS"] = "OFF"
    variables["ALPAKA_CI_SANITIZERS"] = ""
    # cmake install path, if cmake version is not already installed
    variables["ALPAKA_CI_CMAKE_DIR"] = "$HOME/cmake"
    # boost install path, if boost version is not already installed
    variables["BOOST_ROOT"] = "$HOME/boost"
    # required, if boost is compiled during test job
    variables["ALPAKA_CI_BOOST_LIB_DIR"] = "$HOME/boost_libs"
    variables["BOOST_LIBRARYDIR"] = "/opt/boost/${ALPAKA_BOOST_VERSION}/lib"
    # cuda install path, if cuda version is not already installed
    variables["ALPAKA_CI_CUDA_DIR"] = "$HOME/cuda"
    # hip install path, if hip version is not already installed
    variables["ALPAKA_CI_HIP_ROOT_DIR"] = "$HOME/hip"
    # Always enable -Werror in the CI
    variables["alpaka_ENABLE_WERROR"] = "ON"

    ################################################################################################
    ### job dependent environment variables
    ################################################################################################

    variables["CMAKE_BUILD_TYPE"] = job[BUILD_TYPE][VERSION]
    if job[JOB_EXECUTION_TYPE][VERSION] == JOB_EXECUTION_RUNTIME:
        variables["ALPAKA_CI_RUN_TESTS"] = "ON"
    else:
        variables["ALPAKA_CI_RUN_TESTS"] = "OFF"

    variables["ALPAKA_CI_CMAKE_VER"] = job[CMAKE][VERSION]
    variables["ALPAKA_BOOST_VERSION"] = job[BOOST][VERSION]

    # all back-ends are disabled by default
    # back-ends are conditionally enabled depending on the job parameters
    variables[ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE] = "OFF"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE] = "OFF"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE] = "OFF"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE] = "OFF"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE] = "OFF"
    variables[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE] = "OFF"
    variables[ALPAKA_ACC_GPU_CUDA_ENABLE] = "OFF"
    variables["alpaka_ACC_GPU_CUDA_ONLY_MODE"] = "OFF"
    variables[ALPAKA_ACC_GPU_HIP_ENABLE] = "OFF"
    variables["alpaka_ACC_GPU_HIP_ONLY_MODE"] = "OFF"
    variables[ALPAKA_ACC_SYCL_ENABLE] = "OFF"
    # TODO(SimeonEhrig): set libstdc++ for all backends
    # support for different standard c++ libraries is planed
    # https://github.com/alpaka-group/alpaka-job-matrix-library/issues/9
    variables["ALPAKA_CI_STDLIB"] = "libstdc++"
    if job[MDSPAN][VERSION] == ON_VER:
        variables["ALPAKA_TEST_MDSPAN"] = "ON"
    else:
        variables["ALPAKA_TEST_MDSPAN"] = "OFF"

    append_backend_variables(variables, job)

    if job[DEVICE_COMPILER][NAME] == GCC:
        variables["CC"] = "gcc"
        variables["CXX"] = "g++"
        variables["ALPAKA_CI_GCC_VER"] = job[DEVICE_COMPILER][VERSION]
        if (
            ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE in job
            and job[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE][VERSION] == ON_VER
        ):
            variables["ALPAKA_CI_TBB_VERSION"] = "2021.10.0"

    if job[DEVICE_COMPILER][NAME] == CLANG:
        variables["CC"] = "clang"
        variables["CXX"] = "clang++"
        variables["ALPAKA_CI_CLANG_VER"] = job[DEVICE_COMPILER][VERSION]
        if (
            ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE in job
            and job[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE][VERSION] == ON_VER
        ):
            variables["ALPAKA_CI_TBB_VERSION"] = "2021.10.0"

    if job[DEVICE_COMPILER][NAME] == HIPCC:
        variables["CC"] = "clang"
        variables["CXX"] = "clang++"
        variables["CMAKE_HIP_COMPILER"] = "clang++"
        variables["CMAKE_HIP_ARCHITECTURES"] = "${CI_GPU_ARCH}"
        # TODO(SimeonEhrig) check, if we can remove this variable:
        if job[DEVICE_COMPILER][VERSION] == "5.1":
            variables["ALPAKA_CI_CLANG_VER"] = "14"
        elif job[DEVICE_COMPILER][VERSION] == "5.2":
            variables["ALPAKA_CI_CLANG_VER"] = "14"
        elif job[DEVICE_COMPILER][VERSION] == "5.3":
            variables["ALPAKA_CI_CLANG_VER"] = "15"
        elif job[DEVICE_COMPILER][VERSION] == "5.4":
            variables["ALPAKA_CI_CLANG_VER"] = "15"
        elif job[DEVICE_COMPILER][VERSION] == "5.5":
            variables["ALPAKA_CI_CLANG_VER"] = "16"
        elif job[DEVICE_COMPILER][VERSION] == "5.6":
            variables["ALPAKA_CI_CLANG_VER"] = "16"
        elif job[DEVICE_COMPILER][VERSION] == "5.7":
            variables["ALPAKA_CI_CLANG_VER"] = "17"
        elif job[DEVICE_COMPILER][VERSION] == "6.0":
            variables["ALPAKA_CI_CLANG_VER"] = "17"
        else:
            raise RuntimeError(
                "generate_job_yaml.job_variables(): unknown hip version: "
                f"{job[DEVICE_COMPILER][VERSION]}"
            )
        variables["ALPAKA_CI_HIP_VERSION"] = job[DEVICE_COMPILER][VERSION]
        variables["ALPAKA_CI_STDLIB"] = "libstdc++"

    # general configuration, if the CUDA backend is enabled (includes nvcc and clang as CUDA
    # compiler)
    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        variables["ALPAKA_CI_STDLIB"] = "libstdc++"
        variables["CMAKE_CUDA_ARCHITECTURES"] = job[SM_LEVEL][VERSION]
        variables["ALPAKA_CI_CUDA_VERSION"] = job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]

    if job[DEVICE_COMPILER][NAME] == NVCC:
        # general configuration, if nvcc is the CUDA compiler
        variables["CMAKE_CUDA_COMPILER"] = "nvcc"

        # configuration, if GCC is the CUDA host compiler
        if job[HOST_COMPILER][NAME] == GCC:
            variables["CC"] = "gcc"
            variables["CXX"] = "g++"
            variables["ALPAKA_CI_GCC_VER"] = job[HOST_COMPILER][VERSION]
        # configuration, if Clang is the CUDA host compiler
        elif job[HOST_COMPILER][NAME] == CLANG:
            variables["CC"] = "clang"
            variables["CXX"] = "clang++"
            variables["ALPAKA_CI_CLANG_VER"] = job[HOST_COMPILER][VERSION]
        else:
            raise RuntimeError(
                "generate_job_yaml.job_variables(): unknown CUDA host compiler: "
                f"{job[HOST_COMPILER][NAME]}"
            )

    if job[DEVICE_COMPILER][NAME] == CLANG_CUDA:
        variables["CC"] = "clang"
        variables["CXX"] = "clang++"
        variables["ALPAKA_CI_CLANG_VER"] = job[DEVICE_COMPILER][VERSION]
        variables["CMAKE_CUDA_COMPILER"] = "clang++"

    # oneAPI configuration
    if job[DEVICE_COMPILER][NAME] == ICPX:
        variables["CC"] = "icx"
        variables["CXX"] = "icpx"
        if job[DEVICE_COMPILER][VERSION] == "2023.1.0":
            variables["ALPAKA_CI_CLANG_VER"] = "16"
        elif job[DEVICE_COMPILER][VERSION] == "2023.2.0":
            variables["ALPAKA_CI_CLANG_VER"] = "16"
        variables["ALPAKA_CI_STDLIB"] = "libstdc++"
        variables["ALPAKA_CI_ONEAPI_VERSION"] = job[DEVICE_COMPILER][VERSION]
        variables["alpaka_SYCL_ONEAPI_CPU"] = "ON"
        variables["alpaka_SYCL_ONEAPI_CPU_ISA"] = "avx2"

    return variables


@typechecked
def job_tags(job: Dict[str, Tuple[str, str]]) -> List[str]:
    """Add tags to select the correct runner, e.g. CPU only or Nvidia GPU.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.

    Returns:
        List[str]: List of tags.
    """
    if job[JOB_EXECUTION_TYPE][VERSION] == JOB_EXECUTION_COMPILE_ONLY:
        return ["x86_64", "cpuonly"]

    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF_VER
    ):
        return ["x86_64", "cuda"]
    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF_VER
    ):
        return ["x86_64", "rocm"]

    if (
        ALPAKA_ACC_SYCL_ENABLE in job
        and job[ALPAKA_ACC_SYCL_ENABLE][VERSION] != OFF_VER
    ):
        return ["x86_64", "cpuonly"]

    # fallback
    return ["x86_64", "cpuonly"]


def global_variables() -> Dict[str, str]:
    """Generate global variables for the test jobs.

    Returns:
        Dict[str, str]: global variables
    """
    variables: Dict[str, str] = {}

    variables["ALPAKA_CI_OS_NAME"] = "Linux"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE] = "ON"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE] = "ON"
    variables[ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE] = "OFF"
    variables[ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE] = "ON"
    variables[ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE] = "ON"
    variables[ALPAKA_ACC_GPU_CUDA_ENABLE] = "OFF"
    variables["alpaka_ACC_GPU_CUDA_ONLY_MODE"] = "OFF"
    variables[ALPAKA_ACC_GPU_HIP_ENABLE] = "OFF"
    variables["alpaka_ACC_GPU_HIP_ONLY_MODE"] = "OFF"
    variables[ALPAKA_ACC_SYCL_ENABLE] = "OFF"
    # If ALPAKA_CI_ANALYSIS is OFF compile and execute runtime tests else compile only.
    variables["ALPAKA_CI_ANALYSIS"] = "OFF"
    variables["ALPAKA_CI_RUN_TESTS"] = "ON"
    variables["alpaka_CI"] = "GITLAB"
    variables["ALPAKA_CI_SANITIZERS"] = ""
    # TODO(SimeonEhrig) should be set in the before_install.sh and could be removed
    variables["ALPAKA_CI_INSTALL_CUDA"] = "OFF"
    # TODO(SimeonEhrig) should be set in the before_install.sh and could be removed
    variables["ALPAKA_CI_INSTALL_HIP"] = "OFF"
    variables["ALPAKA_CI_CMAKE_DIR"] = "$HOME/cmake"
    variables["BOOST_ROOT"] = "$HOME/boost"
    variables["ALPAKA_CI_BOOST_LIB_DIR"] = "$HOME/boost_libs"
    variables["ALPAKA_CI_CUDA_DIR"] = "$HOME/cuda"
    variables["ALPAKA_CI_HIP_ROOT_DIR"] = "$HOME/hip"
    variables["alpaka_ENABLE_WERROR"] = "ON"

    return variables


@typechecked
def create_job(
    job: Dict[str, Tuple[str, str]], container_version: float, gitlab_images: List[str]
) -> Dict[str, Dict]:
    """Create complete GitLab-CI yaml for a single job

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.
        stage_number (int): Number of the stage. Required for the stage attribute.
        container_version (float): Container version tag.

    Returns:
        Dict[str, Dict]: Job yaml.
    """

    # the job name starts with the device compiler
    job_name = "linux_" + job[DEVICE_COMPILER][NAME] + job[DEVICE_COMPILER][VERSION]
    # if the nvcc is the device compiler, add also the host compiler to the name
    if job[DEVICE_COMPILER][NAME] == NVCC:
        job_name = (
            job_name + "-" + job[HOST_COMPILER][NAME] + job[HOST_COMPILER][VERSION]
        )
    # if Clang-CUDA is the device compiler, add also the CUDA SDK version to the name
    if job[DEVICE_COMPILER][NAME] == CLANG_CUDA:
        job_name = job_name + "-cuda" + job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]

    if job[JOB_EXECUTION_TYPE][VERSION] == JOB_EXECUTION_COMPILE_ONLY:
        job_name += "_compile_only"

    # we need a really explicit job naming, otherwise job names are not unique
    # if a job name is not unique, only one version of the job is executed
    job_name += job_prefix_coding(job)

    job_yaml: Dict = {}
    job_yaml["image"] = job_image(job, container_version, gitlab_images)
    job_yaml["variables"] = job_variables(job)
    job_yaml["script"] = [
        "source ./script/gitlabci/print_env.sh",
        "source ./script/gitlab_ci_run.sh",
    ]
    job_yaml["tags"] = job_tags(job)
    job_yaml["interruptible"] = True

    return {job_name: job_yaml}


@typechecked
def generate_job_yaml_list(
    job_matrix: List[Dict[str, Tuple[str, str]]],
    container_version: float,
    online_check: bool,
) -> List[Dict[str, Dict]]:
    """Generate the job yaml for each job in the job matrix.

    Args:
        job_matrix (List[List[Dict[str, Tuple[str, str]]]]): Job Matrix
        container_version (float): Container version tag.
        online_check (bool): Check if the image exists in the container registry.
            Disabling the check can produce wrong job YAMLs but speed up the
            job generation a lot.

    Returns:
        List[Dict[str, Dict]]: List of GitLab-CI jobs. The key of a dict entry
        is the job name and the value is the body.
    """
    job_matrix_yaml: Dict[str, Dict] = []

    if online_check:
        gitlab_images = get_images_from_registry(str(container_version))
    else:
        # an empty list disables the check
        gitlab_images = []

    for job in job_matrix:
        job_matrix_yaml.append(create_job(job, container_version, gitlab_images))

    return job_matrix_yaml


@typechecked
def distribute_to_waves(
    job_matrix: List[Dict[str, Dict]], wave_size: Dict[str, int] = {}
) -> Dict[str, List[List[Dict[str, Dict]]]]:
    sorted_groups = {}

    for wave in WAVE_GROUP_NAMES:
        sorted_groups[wave] = []

    for job in job_matrix:
        job_name: str = next(iter(job))
        if "_compile_only_" in job_name:
            sorted_groups[JOB_COMPILE_ONLY].append(job)
        elif job_name.startswith("linux_gcc"):
            sorted_groups[JOB_CPU_RUNTIME].append(job)
        # Clang as C++ compiler without CUDA backend
        elif job_name.startswith("linux_clang") and not job_name.startswith(
            "linux_clang-cuda"
        ):
            sorted_groups[JOB_CPU_RUNTIME].append(job)
        elif job_name.startswith("linux_hipcc"):
            # sorted_groups[JOB_ROCM_RUNTIME].append(job)
            sorted_groups[JOB_RUNTIME].append(job)
        elif job_name.startswith("linux_nvcc") and "gcc" in job_name:
            # sorted_groups[JOB_NVCC_GCC_RUNTIME].append(job)
            sorted_groups[JOB_RUNTIME].append(job)
        elif job_name.startswith("linux_nvcc") and "clang" in job_name:
            # sorted_groups[JOB_NVCC_CLANG_RUNTIME].append(job)
            sorted_groups[JOB_RUNTIME].append(job)
        elif job_name.startswith("linux_clang-cuda"):
            # sorted_groups[JOB_CLANG_CUDA_RUNTIME].append(job)
            sorted_groups[JOB_RUNTIME].append(job)
        elif job_name.startswith("linux_icpx"):
            # sorted_groups[JOB_ICPX_RUNTIME].append(job)
            sorted_groups[JOB_RUNTIME].append(job)
        else:
            sorted_groups[JOB_UNKNOWN].append(job)

    for wave in WAVE_GROUP_NAMES:
        if not wave in wave_size:
            wave_size[wave] = len(sorted_groups[wave])
        else:
            # if max_jobs is negative, set to 0
            if wave_size[wave] < 0:
                wave_size[wave] = 0
            # if max_jobs is greater than len(sorted_groups[wave]), crop it to len(sorted_groups[wave])
            elif wave_size[wave] > len(sorted_groups[wave]):
                wave_size[wave] = len(sorted_groups[wave])

    wave_matrix: Dict[str : List[List[Dict[str, Dict]]]] = {}

    for wave in WAVE_GROUP_NAMES:
        wave_matrix[wave] = []

        # handle special case: range() expect step size bigger than 1
        wave_size[wave] = 1 if wave_size[wave] == 0 else wave_size[wave]

        for i in range(0, len(sorted_groups[wave]), wave_size[wave]):
            wave_matrix[wave].append(sorted_groups[wave][i : i + wave_size[wave]])

    return wave_matrix


@typechecked
def write_job_yaml(
    job_matrix: Dict[str, List[List[Dict[str, Dict]]]],
    path: str,
):
    """Write GitLab-CI jobs to file.

    Args:
        job_matrix (List[List[Dict[str, Dict]]]): List of GitLab-CI jobs. The
        key of a dict entry is the job name and the value is the body.
        path (str): Path of the GitLab-CI yaml file.
    """
    with open(path, "w", encoding="utf-8") as output_file:
        # If there is no CI job, create a dummy job.
        # This can happen if the filter filters out all jobs.
        number_of_jobs = 0
        for wave_name in WAVE_GROUP_NAMES:
            number_of_jobs += len(job_matrix[wave_name])

        if number_of_jobs == 0:
            yaml.dump(
                {
                    "dummy-job": {
                        "image": "alpine:latest",
                        "interruptible": True,
                        "script": [
                            'echo "This is a dummy job so that the CI does not fail."'
                        ],
                    }
                },
                output_file,
            )
            return

        stages: Dict[str, List[str]] = {"stages": []}

        for wave_name in WAVE_GROUP_NAMES:
            # setup all stages
            for stage_number in range(len(job_matrix[wave_name])):
                stages["stages"].append(f"{wave_name}-stage{stage_number}")

        yaml.dump(stages, output_file)
        output_file.write("\n")
        # add global variables for all jobs
        yaml.dump({"variables": global_variables()}, output_file)
        output_file.write("\n")

        # TODO: remove me, when all cuda and hip custom jobs are generated by the job generator
        # The CUDA and HIP jobs inherent from a job template written in yaml
        script_path = os.path.abspath(__file__)
        with open(
            os.path.abspath(
                os.path.join(os.path.dirname(script_path), "../gitlabci/job_base.yml")
            ),
            "r",
            encoding="utf8",
        ) as file:
            job_base_yaml = yaml.load(file, yaml.loader.SafeLoader)
        yaml.dump(job_base_yaml, output_file)

        for wave_name in WAVE_GROUP_NAMES:
            # Writes each job separately to the file.
            # If all jobs would be collected first in dict, the order would be not guarantied.
            for stage_number, wave in enumerate(job_matrix[wave_name]):
                # Improve the readability of the generated job yaml
                output_file.write(
                    f"# <<<<<<<<<<<<< {wave_name}-stage{stage_number} >>>>>>>>>>>>>\n\n"
                )
                for job in wave:
                    # the first key is the name
                    job[list(job.keys())[0]][
                        "stage"
                    ] = f"{wave_name}-stage{stage_number}"

                    yaml.dump(job, output_file)
                    output_file.write("\n")
