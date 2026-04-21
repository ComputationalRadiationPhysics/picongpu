"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from pathlib import Path
from unittest import TestCase

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


class TestMinimal(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_has_finished_run(self):
        with (self.result_path / "simOutput" / "output").open("r") as file:
            assert "full simulation time:" in file.read()
