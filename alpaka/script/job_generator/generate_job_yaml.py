"""Create GitLab-CI job description written in yaml from the job matrix."""

from typing import List, Dict, Tuple
from typeguard import typechecked
import os, yaml
from packaging import version as pk_version


from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


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

    return version_str


@typechecked
def job_image(job: Dict[str, Tuple[str, str]], container_version: float) -> str:
    """Generate the image url deppending on the host and device compiler and the selected backend
    of the job.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.
        container_version (float): Container version tag.

    Returns:
        str: Full container url, which can be used with docker pull.
    """
    container_url = "registry.hzdr.de/crp/alpaka-group-container/"
    container_url += "alpaka-ci-ubuntu" + job[UBUNTU][VERSION]

    # If only the GCC is used, use special gcc version of the container.
    if job[HOST_COMPILER][NAME] == GCC and job[DEVICE_COMPILER][NAME] == GCC:
        container_url += "-gcc"

    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF
    ):
        # Cast cuda version shape. E.g. from 11.0 to 110
        container_url += "-cuda" + str(
            int(float(job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION]) * 10)
        )
        if job[HOST_COMPILER][NAME] == GCC:
            container_url += "-gcc"

    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF
    ):
        container_url += "-rocm" + job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION]

    # append container tag
    container_url += ":" + str(container_version)
    return container_url


@typechecked
def job_variables(job: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Add variables to the job depending of the job dict.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict

    Returns:
        Dict[str, str]: Dict of {variable name : variable value}.
    """
    variables: Dict[str, str] = {}

    return variables


@typechecked
def job_tags(job: Dict[str, Tuple[str, str]]) -> List[str]:
    """Add tags to select the correct runner, e.g. CPU only or Nvidia GPU.

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.

    Returns:
        List[str]: List of tags.
    """
    if (
        ALPAKA_ACC_GPU_CUDA_ENABLE in job
        and job[ALPAKA_ACC_GPU_CUDA_ENABLE][VERSION] != OFF
    ):
        return ["x86_64", "cuda"]
    if (
        ALPAKA_ACC_GPU_HIP_ENABLE in job
        and job[ALPAKA_ACC_GPU_HIP_ENABLE][VERSION] != OFF
    ):
        return ["x86_64", "rocm"]
    return ["x86_64", "cpuonly"]


def global_variables() -> Dict[str, str]:
    """Generate global variables for the test jobs.

    Returns:
        Dict[str, str]: global variables
    """
    variables: Dict[str, str] = {}

    variables["ALPAKA_GITLAB_CI_CONTAINER_VERSION"] = "1.4"
    variables["ALPAKA_CI_OS_NAME"] = "Linux"
    variables["alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE"] = "ON"
    variables["alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE"] = "ON"
    variables["alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE"] = "OFF"
    variables["alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE"] = "OFF"
    variables["alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE"] = "ON"
    variables["alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE"] = "ON"
    variables["alpaka_ACC_ANY_BT_OMP5_ENABLE"] = "OFF"
    variables["alpaka_ACC_ANY_BT_OACC_ENABLE"] = "OFF"
    variables["alpaka_ACC_GPU_CUDA_ENABLE"] = "OFF"
    variables["alpaka_ACC_GPU_CUDA_ONLY_MODE"] = "OFF"
    variables["alpaka_ACC_GPU_HIP_ENABLE"] = "OFF"
    variables["alpaka_ACC_GPU_HIP_ONLY_MODE"] = "OFF"
    # If ALPAKA_CI_ANALYSIS is OFF compile and execute runtime tests else compile only.
    variables["ALPAKA_CI_ANALYSIS"] = "OFF"
    variables["ALPAKA_CI_RUN_TESTS"] = "ON"
    variables["alpaka_CI"] = "GITLAB"
    # needs to be enabled, that test on the GPU are executed
    variables["ALPAKA_FORCE_RUNTIME_TEST"] = "ON"
    variables["ALPAKA_CI_SANITIZERS"] = ""
    variables["ALPAKA_CI_INSTALL_CUDA"] = "OFF"
    variables["ALPAKA_CI_INSTALL_HIP"] = "OFF"
    variables["ALPAKA_CI_CMAKE_DIR"] = "$HOME/cmake"
    variables["BOOST_ROOT"] = "$HOME/boost"
    variables["ALPAKA_CI_BOOST_LIB_DIR"] = "$HOME/boost_libs"
    variables["ALPAKA_CI_CUDA_DIR"] = "$HOME/cuda"
    variables["ALPAKA_CI_HIP_ROOT_DIR"] = "$HOME/hip"

    return variables


@typechecked
def create_job(
    job: Dict[str, Tuple[str, str]], container_version: float
) -> Dict[str, Dict]:
    """Create complete GitLab-CI yaml for a single job

    Args:
        job (Dict[str, Tuple[str, str]]): Job dict.
        stage_number (int): Number of the stage. Required for the stage attribute.
        container_version (float): Container version tag.

    Returns:
        Dict[str, Dict]: Job yaml.
    """

    # TODO: implement me!

    return {}


@typechecked
def generate_job_yaml_list(
    job_matrix: List[Dict[str, Tuple[str, str]]],
    container_version: float,
) -> List[Dict[str, Dict]]:
    """Generate the job yaml for each job in the job matrix.

    Args:
        job_matrix (List[List[Dict[str, Tuple[str, str]]]]): Job Matrix
        container_version (float): Container version tag.

    Returns:
        List[Dict[str, Dict]]: List of GitLab-CI jobs. The key of a dict entry
        is the job name and the value is the body.
    """
    job_matrix_yaml: Dict[str, Dict] = []
    for job in job_matrix:
        job_matrix_yaml.append(create_job(job, container_version))

    return job_matrix_yaml


@typechecked
def write_job_yaml(
    job_matrix: List[List[Dict[str, Dict]]],
    path: str,
):
    """Write GitLab-CI jobs to file.

    Args:
        job_matrix (List[List[Dict[str, Dict]]]): List of GitLab-CI jobs. The
        key of a dict entry is the job name and the value is the body.
        path (str): Path of the GitLab-CI yaml file.
    """
    with open(path, "w", encoding="utf-8") as output_file:
        # setup all stages
        stages: Dict[str, List[str]] = {"stages": []}
        for stage_number in range(len(job_matrix)):
            stages["stages"].append(f"stage{stage_number}")
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

        # Writes each job separately to the file.
        # If all jobs would be collected first in dict, the order would be not guarantied.
        for stage_number, wave in enumerate(job_matrix):
            # Improve the readability of the generated job yaml
            output_file.write(f"# <<<<<<<<<<<<< stage {stage_number} >>>>>>>>>>>>>\n\n")
            for job in wave:
                # the first key is the name
                job[list(job.keys())[0]]["stage"] = "stage" + str(stage_number)

                yaml.dump(job, output_file)
                output_file.write("\n")
