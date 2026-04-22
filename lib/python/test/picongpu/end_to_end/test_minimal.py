"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from pathlib import Path
from subprocess import run
from time import sleep
from unittest import TestCase
from psutil import pid_exists

from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation

from .arbitrary_parameters import NUMBER_OF_CELLS, UPPER_BOUNDARY

logging.basicConfig(level=logging.INFO)


def basic_simulation():
    return Simulation(
        max_steps=0,
        solver=ElectromagneticSolver(
            method="Yee",
            cfl=1.0,
            grid=Cartesian3DGrid(
                number_of_cells=NUMBER_OF_CELLS,
                lower_bound=[0, 0, 0],
                # cell size is slightly different from 1
                upper_bound=UPPER_BOUNDARY,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
        ),
    )


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(0, jobs=20)
    return sim


SIM = None
TIMEOUT_COUNT = 10


class TestMinimal(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM
        self._gather_results()

    def _gather_results(self):
        # This currently only handles the case of a local process running:
        with (self.result_path / "submission_information.txt").open("r") as file:
            pid = int(file.read())
        for _ in range(TIMEOUT_COUNT):
            if pid_exists(pid):
                sleep(5)
            else:
                return run([self.result_path / "link_results.sh", self.result_path])
        raise Exception("Simulation is still running after {num_attempts=}.")

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_has_finished_run(self):
        with (self.result_path / "simOutput" / "output").open("r") as file:
            assert "full simulation time:" in file.read()
