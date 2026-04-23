"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
from subprocess import run
from time import sleep

import numpy as np
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
        if function():
            sleep(sleep_interval)
        else:
            return
    raise Exception("Waiting for function did not return after {num_attempts=}.")


def _make_wait_function_from(submission_information, submission_system="bash"):
    # This currently only handles the case of a local process running.
    # For other submission systems, this would have to
    # - parse the corresponding information from the submission_information.txt
    # - and query the corresponding batch system.
    if submission_system == "bash":
        with submission_information.open("r") as file:
            pid = int(file.read())
        return lambda: pid_exists(pid)
    raise NotImplementedError("Only bash submission information can be parsed at this point.")


def gather_results(result_path: Path):
    # job has to finish
    _wait_until(_make_wait_function_from(result_path / "submission_information.txt", "bash"))
    # CWLtool has to copy over the files
    _wait_until(lambda: (result_path / "link_results.sh").exists())

    run([result_path / "link_results.sh", result_path])
