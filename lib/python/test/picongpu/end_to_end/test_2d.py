"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Full end-to-end test of a 2D (2D3V) PICMI setup.

Builds and runs a minimal 2D laser-injection benchmark (a port of the
``IncidentField`` example). Running this test also serves as the compile check
for the 2D setup: the E2E build phase must succeed for a 2D3V configuration
(dimension.param = DIM2, 2-component grid, 2D laser).
"""

import logging
from pathlib import Path
from unittest import TestCase

from picongpu import rc_params
from picongpu.picmi import (
    Cartesian2DGrid,
    ElectromagneticSolver,
    PseudoRandomLayout,
    Simulation,
    Species,
    UniformDistribution,
)
from picongpu.picmi.constants import c
from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec
from picongpu.picmi.lasers import GaussianLaser, PolarizationType

from .arbitrary_parameters import directory_in_home, gather_results

logging.basicConfig(level=logging.INFO)

NUMBER_OF_CELLS = [128, 128]
CELL_SIZE = 0.5e-3


def basic_simulation():
    grid = Cartesian2DGrid(
        number_of_cells=NUMBER_OF_CELLS,
        lower_bound=[0, 0],
        upper_bound=[NUMBER_OF_CELLS[0] * CELL_SIZE, NUMBER_OF_CELLS[1] * CELL_SIZE],
        lower_boundary_conditions=["open", "open"],
        upper_boundary_conditions=["open", "open"],
    )
    solver = ElectromagneticSolver(method="Yee", cfl=0.999, grid=grid)
    sim = Simulation(max_steps=0, solver=solver)

    electrons = Species(
        particle_type="electron",
        name="electrons",
        initial_distribution=UniformDistribution(density=1.0e24),
    )
    sim.add_species(electrons, layout=PseudoRandomLayout(n_macroparticles_per_cell=2))

    laser_duration = 26.0e-15
    pulse_init = 15.0
    laser = GaussianLaser(
        wavelength=800e-9,
        waist=16e-3,
        duration=laser_duration,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        focal_position=[NUMBER_OF_CELLS[0] * CELL_SIZE / 2.0, NUMBER_OF_CELLS[1] * CELL_SIZE / 2.0, 0.0],
        centroid_position=[
            NUMBER_OF_CELLS[0] * CELL_SIZE / 2.0,
            -0.5 * pulse_init * laser_duration * c,
            0.0,
        ],
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    )
    sim.add_laser(laser, None)
    sim.diagnostics = [Checkpoint(period=TimeStepSpec[::100])]
    return sim


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    if "rosi-hzdr" in rc_params.get("preset", "bash"):
        sim.picongpu_get_runner().setup_dir = directory_in_home() / "setup"
        sim.picongpu_get_runner().run_dir = directory_in_home() / "run"
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(0)
    return sim


SIM = None


class Test2D(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
            self.sim = SIM
            gather_results(self.result_path)
        self.sim = SIM

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_has_finished_run(self):
        with (self.result_path / "simOutput" / "output").open("r") as file:
            assert "full simulation time:" in file.read()
