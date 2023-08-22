"""Copyright 2023 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Add custom jobs. For example loaded from a yaml file."""

from genericpath import isfile
import os, yaml
from typing import List, Dict, Callable
from typeguard import typechecked


@typechecked
def read_jobs_from_folder(
    path: str, filter: Callable = lambda name: True
) -> List[Dict[str, Dict]]:
    """Read all job descriptions from the files located in a specific folder.
    The function ignore sub folders.

    Args:
        path (str): Path of the folder, where the job files are located
        filter (Callable, optional): Filter function, which takes the filename as argument. If the
        function returns False, the file is ignored.

    Returns:
        List[Dict[str, Dict]]: List of GitLab CI jobs
    """
    if not os.path.exists(path):
        print(f"\033[31mERROR: {path} does not exists\033[m")
        exit(1)
    if not os.listdir(path):
        print(f"\033[33mWARNING: {path} is empty\033[m")
        return []

    custom_job_list: List[Dict[str, Dict]] = []

    for file_name in os.listdir(path):
        abs_file_path = os.path.join(path, file_name)
        if os.path.isfile(abs_file_path) and filter(file_name):
            with open(abs_file_path, "r", encoding="utf8") as job_yaml:
                for job_name, job_body in yaml.load(
                    job_yaml, yaml.loader.SafeLoader
                ).items():
                    custom_job_list.append({job_name: job_body})

    return custom_job_list


@typechecked
def add_custom_jobs(job_matrix_yaml: List[Dict[str, Dict]], container_version: float):
    """Read custom jobs from yaml files and add it to the job_matrix_yaml.

    Args:
        job_matrix_yaml (List[Dict[str, Dict]]): The job matrix, containing the yaml code
        for each job.
        container_version (float): Used container version.

    Raises:
        RuntimeError: Throw error, if yaml file of custom jobs does not exits.
    """
    # load custom jobs from the folder script/gitlabci
    script_gitlab_ci_folder = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../gitlabci/")
    )

    for path in [script_gitlab_ci_folder]:
        job_matrix_yaml += read_jobs_from_folder(
            path,
            # filter file names
            lambda name: name != "job_base.yml"
            and name.startswith("job_")
            and name.endswith(".yml"),
        )
