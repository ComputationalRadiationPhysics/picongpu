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


def gather_results(result_path: Path):
    # This currently only handles the case of a local process running:
    with (result_path / "submission_information.txt").open("r") as file:
        pid = int(file.read())
    for _ in range(TIMEOUT_COUNT):
        if pid_exists(pid):
            sleep(5)
        else:
            return run([result_path / "link_results.sh", result_path])
    raise Exception("Simulation is still running after {num_attempts=}.")
