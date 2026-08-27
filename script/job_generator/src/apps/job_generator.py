"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate CI jobs for alpaka.
"""

import sys
import os
import argparse
import random
from itertools import chain
import bashi
import alpaka_bashi


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
        help="Display combination list.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter the jobs with a Python regex that checks the job names.",
    )

    parser.add_argument(
        "--split-pipeline",
        action="store_true",
        help="Write job pipelines in separate output files.",
    )

    for wave_name in alpaka_bashi.globals.CI_PIPELINE_NAME_MAPPING:
        parser.add_argument(
            f"--pipeline-out-{wave_name}",
            type=str,
            required="--split-pipeline" in sys.argv,
            # add `all` and remove `JOB_UNKNOWN` from the choices
            help=f"Output path of the job yaml for the pipeline {wave_name}",
        )

    parser.add_argument(
        "--no-image-check",
        action="store_false",
        help="Disable registry check for existing Docker image.",
    )

    parser.add_argument(
        "--debug-print",
        type=bashi.FilterDebugMode,
        choices=list(bashi.FilterDebugMode),
        default="off",
        help="Display Indicate which combinations passed through the filter chain and which did "
        "not.Green text indicates that the combination passed through the filter chain; red text"
        " indicates that it did not. Add the keyword `passed` if colored output is not available."
        "Option `normal` is easy human readable output. If `args` is set, the output can be "
        "directly passed to the validator",
    )

    return parser.parse_args()


def setup_row_printer() -> None:
    """Set extra configurations for the bashi.print_row_nice() function"""
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.BUILD_TYPE, "buildType")
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.JOB_EXECUTION_TYPE, "jobType")

    for val_name, aliases in alpaka_bashi.get_version_aliases().items():
        bashi.add_print_row_nice_version_alias(val_name, aliases)


def get_filter_name(args: argparse.Namespace) -> str:
    """Return filter string CI jobs. All jobs, which does not match the filter regex, will be
    removed.

    Ether the filter is set via command line argument --filter or via Git commit message with the
    prefix `CI_FILTER:`.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        str: The filter regex. Return empty string, if no filter was set.
    """
    commit_message_filter_prefix = "CI_FILTER:"
    if os.getenv("CI_COMMIT_MESSAGE"):
        for line in os.getenv("CI_COMMIT_MESSAGE", "").split("\n"):
            striped_line = line.strip()
            if striped_line.strip().startswith(commit_message_filter_prefix):
                return striped_line[len(commit_message_filter_prefix) :].strip()

    if args.filter:
        return args.filter

    return ""


def write_single_file_job_configuration(
    pipelines: dict[bashi.ValueVersion, bashi.CombinationList], args: argparse.Namespace
):
    """Write generated GitLab CI yaml code to stdout.

    Args:
        pipelines (dict[bashi.ValueVersion, bashi.CombinationList]): All CI pipelines and their
            jobs.
        args (argparse.Namespace): Application arguments.
    """
    job_filter_name = get_filter_name(args)

    jobs = alpaka_bashi.get_job_configuration(
        combination_list=list(chain(*pipelines.values())),
        container_version=str(args.version),
        image_check=args.no_image_check,
        stages=False,
        wave_sizes=None,
    )
    jobs |= alpaka_bashi.get_special_jobs(
        container_version=str(args.version),
        image_check=args.no_image_check,
        stage_name="",
        job_filter=job_filter_name,
    )
    if len(jobs) == 0:
        jobs = alpaka_bashi.get_dummy_job_yaml()

    alpaka_bashi.write_job_yaml(jobs, sys.stdout)


def write_multiple_file_job_configuration(
    pipelines: dict[bashi.ValueVersion, bashi.CombinationList],
    wave_sizes: dict[bashi.ValueVersion, int],
    args: argparse.Namespace,
):
    """Write generated GitLab CI yaml code to different files.

    Args:
        pipelines (dict[bashi.ValueVersion, bashi.CombinationList]): All CI pipelines and their
        jobs.
        wave_sizes (dict[bashi.ValueVersion, int]): Size of each wave.
        args (argparse.Namespace): Application arguments.
    """
    job_filter_name = get_filter_name(args)

    for pipeline_ver, combinations in pipelines.items():
        pipeline_name = alpaka_bashi.get_version_aliases()[alpaka_bashi.globals.CI_PIPELINE_NAME][
            pipeline_ver
        ]
        output_path = getattr(args, f"pipeline-out-{pipeline_name}".replace("-", "_"))

        if pipeline_name == alpaka_bashi.CI_PIPELINE_SPECIAL:
            jobs = alpaka_bashi.get_special_jobs(
                container_version=str(args.version),
                image_check=args.no_image_check,
                stage_name=pipeline_name,
                job_filter=job_filter_name,
            )
        else:
            jobs = alpaka_bashi.get_job_configuration(
                combination_list=combinations,
                container_version=str(args.version),
                image_check=args.no_image_check,
                stages=True,
                wave_sizes=wave_sizes,
            )

        if len(jobs) == 0:
            jobs = alpaka_bashi.get_dummy_job_yaml(pipeline_name)

        with open(output_path, "w", encoding="utf-8") as output_file:
            alpaka_bashi.write_job_yaml(jobs, output_file)


def main() -> None:
    """Entry point"""
    args = get_args()

    setup_row_printer()

    software_versions = alpaka_bashi.get_software_versions_for_alpaka()

    param_matrix: bashi.ParameterValueMatrix = bashi.get_parameter_value_matrix(
        software_versions=software_versions
    )

    version_relation = alpaka_bashi.get_alpaka_version_relation()

    alpaka_filter = alpaka_bashi.AlpakaFilter()
    runtime_infos = bashi.get_runtime_infos(param_matrix, version_relation)
    runtime_infos |= alpaka_bashi.get_runtime_infos(software_versions, version_relation)

    comb_list: bashi.CombinationList = bashi.generate_combination_list(
        parameter_value_matrix=param_matrix,
        runtime_infos=runtime_infos,
        custom_filter=alpaka_filter,
        version_relation=version_relation,
        # change me to display which combinations passed and did not pass the filter chain
        debug_print=args.debug_print,
    )
    print(f"number of combinations: {len(comb_list)}", file=sys.stderr)

    comb_list = alpaka_bashi.add_combinations_parameters(comb_list)

    if not alpaka_bashi.verify(comb_list, param_matrix, version_relation, runtime_infos):
        print("ERROR: Result is incorrect", file=sys.stderr)
        sys.exit(1)

    print("Result is correct", file=sys.stderr)

    job_filter_name = get_filter_name(args)
    if job_filter_name:
        comb_list = alpaka_bashi.filter_combinations(comb_list, job_filter_name)
        print(f"number of filtered combinations: {len(comb_list)}", file=sys.stderr)

    if args.print_combinations:
        for c in comb_list:
            bashi.print_row_nice(c)
        sys.exit(0)

    # shuffle jobs to increase the chance to run different compiler in the first wave
    random.Random(42).shuffle(comb_list)

    pipelines = alpaka_bashi.distribute_to_pipelines(comb_list)

    # If the pipelines are not splitted and therefore written to different files, write everything
    # to stdout.
    # We split up the pipelines and merge again, because in the meantime reorder operations can be
    # applied on the different pipelines.
    # By the way, it also automatically sort the jobs by pipeline.
    if not args.split_pipeline:
        write_single_file_job_configuration(pipelines, args)
    else:
        wave_sizes = {
            alpaka_bashi.CI_PIPELINE_COMPILE_ONLY_VER: 30,
            alpaka_bashi.CI_PIPELINE_RUNTIME_CPU_VER: 30,
        }

        write_multiple_file_job_configuration(pipelines, wave_sizes, args)

    sys.exit(0)


if __name__ == "__main__":
    main()
