# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import environ
from pathlib import Path
from subprocess import run
from time import sleep

import numpy as np
from picongpu import rc_params
from picongpu.pypicongpu.util import alt
from psutil import pid_exists

NUMBER_OF_CELLS = [64, 64, 32]
LOWER_BOUNDARY = np.array([0.0, 0.0, 0.0])
UPPER_BOUNDARY = np.array([64.0, 66.0, 74.0])
CELL_SIZE = UPPER_BOUNDARY / NUMBER_OF_CELLS

ALL_ORIGINS = ["total", "global", "local", "moving_window", "local_with_guards"]
ALL_ORIGINS_WITHOUT_GUARDS = [origin for origin in ALL_ORIGINS if not origin.endswith("guards")]
ALL_PRECISIONS = ["cell", "sub_cell"]
ALL_UNITS = ["cell", "si", "pic"]
NUMBER_OF_GUARD_CELLS = [8, 8, 4]
EPSILON = 1.0e-5

TIMEOUT_COUNT = 100


def _wait_until(function, sleep_interval=5, timeout_count=TIMEOUT_COUNT):
    for _ in range(timeout_count):
        if not function():
            sleep(sleep_interval)
        else:
            return
    raise Exception(f"Waiting for function did not return after {timeout_count=} attempts.")


def _parse_submission_system(submission_cmd):
    submission_cmd = submission_cmd.strip()
    if submission_cmd.startswith("s"):
        return "slurm"
    if submission_cmd.startswith("bash"):
        return "bash"
    if submission_cmd.startswith("zsh"):
        return "zsh"
    raise NotImplementedError(f"Unable to parse {submission_cmd=} into a submission system.")


def _is_finished_slurm_status(status_str):
    match status_str:
        case "COMPLETED":
            return True
        case str(cancelled_or_failed) if cancelled_or_failed == "FAILED" or cancelled_or_failed.startswith("CANCELLED"):
            raise RuntimeError(f"Slurm job did not complete successfully but with status {status_str=}.")
        case "PENDING" | "RUNNING":
            return False
    raise NotImplementedError(f"Unable to parse {status_str=} into slurm status.")


def _make_wait_function_from(submission_information, submission_cmd="bash"):
    with submission_information.open("r") as file:
        pid = int(file.read().split()[-1])
    match _parse_submission_system(submission_cmd):
        case "bash" | "zsh":
            return lambda: not pid_exists(pid)
        case "slurm":
            return lambda: any(
                map(
                    lambda s: s.startswith(str(pid)) and _is_finished_slurm_status(s.split("|")[1]),
                    # There are alternatives to running a subprocess ourselves but they aren't better at the time of writing:
                    # - pyslurm: Links against slurm development libraries which are not necessarily installed on a cluster.
                    # - simple_slurm: Just wraps the subprocess.run calls and doesn't have the full interface implemented.
                    run(["sacct", "-S", "2026-01-01", "-XPno", "jobid,state"], capture_output=True)
                    .stdout.decode()
                    .split("\n"),
                )
            )
        case _:
            raise NotImplementedError("Only bash submission information can be parsed at this point.")


def gather_results(result_path: Path):
    # job has to finish
    _wait_until(
        _make_wait_function_from(result_path / "submission_information.txt", rc_params.get("tbg_submit", "bash"))
    )
    # CWLtool has to copy over the files
    _wait_until(lambda: (result_path / "link_results.sh").exists())

    run([result_path / "link_results.sh", result_path])


def directory_in(path, offset=0):
    if not isinstance(path, Path):
        return directory_in(Path(path))
    new_offset = (
        max([offset, *map(lambda p: alt(lambda: int(str(p.name)), 0, ignore=(ValueError,)), path.glob("*"))]) + 1
    )
    directory = path / f"{new_offset:>06}"
    return directory if not directory.exists() else directory_in(path, offset=new_offset)


def directory_in_home():
    return directory_in(Path(environ["HOME"]) / "data")
