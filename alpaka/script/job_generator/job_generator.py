"""Copyright 2023 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate GitLab-CI test jobs yaml for the vikunja CI."""

import argparse
import sys
import os
import random
from typing import List, Dict, Tuple
from collections import OrderedDict

import alpaka_job_coverage as ajc
from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import filter_job_list, reorder_job_list

from versions import (
    get_sw_tuple_list,
    get_compiler_versions,
    get_backend_matrix,
)
from alpaka_filter import alpaka_post_filter
from custom_job import add_custom_jobs
from reorder_jobs import reorder_jobs
from generate_job_yaml import (
    generate_job_yaml_list,
    write_job_yaml,
    distribute_to_waves,
    JOB_COMPILE_ONLY,
    JOB_UNKNOWN,
    WAVE_GROUP_NAMES,
)
from job_modifier import add_job_parameters
from verify import verify, verify_parameters
from util import print_warn


def get_args() -> argparse.Namespace:
    """Define and parse the commandline arguments.

    Returns:
        argparse.Namespace: The commandline arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate job matrix and create GitLab CI .yml.")

    parser.add_argument("version", type=float, help="Version number of the used CI container.")
    parser.add_argument(
        "--print-combinations",
        action="store_true",
        help="Display combination matrix.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify generated combination matrix")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Combine flags: --print-combinations and --verify",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="./jobs.yml",
        help="Path of the generated jobs yaml.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter the jobs with a Python regex that checks the job names.",
    )

    parser.add_argument(
        "--reorder",
        type=str,
        default="",
        help="Orders jobs by their names. Expects a string consisting of one or more Python regex. "
        'The regex are separated by whitespaces. For example, the regex "^NVCC ^GCC" has the '
        "behavior that all NVCC jobs are executed first and then all GCC jobs.",
    )

    parser.add_argument(
        "--wave",
        type=str,
        # add `all` and remove `JOB_UNKNOWN` from the choices
        choices=["all"] + list(set(WAVE_GROUP_NAMES) - set([JOB_UNKNOWN])),
        default="all",
        help="Generates only jobs for a specific wave group.",
    )

    parser.add_argument(
        "--no-image-check",
        action="store_false",
        help="Disable registry check for existing Docker image.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # setup the parameters
    parameters: OrderedDict = OrderedDict()
    enable_clang_cuda = True
    parameters[HOST_COMPILER] = get_compiler_versions(clang_cuda=enable_clang_cuda)
    parameters[DEVICE_COMPILER] = get_compiler_versions(clang_cuda=enable_clang_cuda)
    parameters[BACKENDS] = get_backend_matrix()
    parameters[CMAKE] = get_sw_tuple_list(CMAKE)
    parameters[BOOST] = get_sw_tuple_list(BOOST)
    parameters[UBUNTU] = get_sw_tuple_list(UBUNTU)
    parameters[CXX_STANDARD] = get_sw_tuple_list(CXX_STANDARD)
    parameters[BUILD_TYPE] = get_sw_tuple_list(BUILD_TYPE)
    parameters[JOB_EXECUTION_TYPE] = get_sw_tuple_list(JOB_EXECUTION_TYPE)
    parameters[MDSPAN] = get_sw_tuple_list(MDSPAN)

    # print a warning, if a parameter value is not supported by the ajc-library
    verify_parameters(parameters)

    job_matrix: List[Dict[str, Tuple[str, str]]] = ajc.create_job_list(
        parameters=parameters,
        post_filter=alpaka_post_filter,
        pair_size=2,
    )

    add_job_parameters(job_matrix)

    if args.print_combinations or args.all:
        print(f"number of combinations before reorder: {len(job_matrix)}")

    ajc.shuffle_job_matrix(job_matrix)
    reorder_jobs(job_matrix)

    if args.print_combinations or args.all:
        for compiler in job_matrix:
            print(compiler)

        print(f"number of combinations: {len(job_matrix)}")

    if args.verify or args.all:
        if not verify(job_matrix):
            sys.exit(1)

    job_matrix_yaml = generate_job_yaml_list(
        job_matrix=job_matrix,
        container_version=args.version,
        online_check=args.no_image_check,
    )

    add_custom_jobs(job_matrix_yaml, args.version)
    # shuffle jobs to better utilize the special runner
    random.Random(42).shuffle(job_matrix_yaml)

    filter_regix = args.filter
    reorder_regix = args.reorder

    COMMIT_MESSAGE_FILTER_PREFIX = "CI_FILTER:"
    COMMIT_MESSAGE_REORDER_PREFIX = "CI_REORDER:"

    # If the environment variable CI_COMMIT_MESSAGE exists (like in GitLabCI Job),
    # check for the prefixes and overwrite argument values of --filter and --reorder if
    # prefix was found in the beginning of a line.
    if os.getenv("CI_COMMIT_MESSAGE"):
        for line in os.getenv("CI_COMMIT_MESSAGE").split("\n"):
            striped_line = line.strip()
            if striped_line.strip().startswith(COMMIT_MESSAGE_FILTER_PREFIX):
                filter_regix = striped_line[len(COMMIT_MESSAGE_FILTER_PREFIX) :].strip()
            if striped_line.startswith(COMMIT_MESSAGE_REORDER_PREFIX):
                reorder_regix = striped_line[len(COMMIT_MESSAGE_REORDER_PREFIX) :].strip()

    if filter_regix:
        job_matrix_yaml = filter_job_list(job_matrix_yaml, filter_regix)

    if reorder_regix:
        job_matrix_yaml = reorder_job_list(job_matrix_yaml, reorder_regix)

    # wave_job_matrix = ajc.distribute_to_waves(job_matrix_yaml, 10)
    wave_job_matrix = distribute_to_waves(job_matrix_yaml, {JOB_COMPILE_ONLY: 20})

    if wave_job_matrix[JOB_UNKNOWN]:
        print_warn('Generator distributed jobs of type "JOB_UNKNOWN"')
        for wave in wave_job_matrix[JOB_UNKNOWN]:
            for job in wave:
                print(job)

    if args.wave != "all":
        filter_wave_name = args.wave
        wave_job_matrix = {filter_wave_name: wave_job_matrix[filter_wave_name]}
        for wave_name in WAVE_GROUP_NAMES:
            if wave_name not in wave_job_matrix:
                wave_job_matrix[wave_name] = []

    write_job_yaml(
        job_matrix=wave_job_matrix,
        path=args.output_path,
    )
