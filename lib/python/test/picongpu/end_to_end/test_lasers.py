"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from functools import reduce
from pathlib import Path
from unittest import TestCase

import numpy as np
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GaussianLaser,
    PlaneWaveLaser,
    Simulation,
    constants,
)
from picongpu.picmi import Species as Species
from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec
from picongpu.picmi.lasers import PolarizationType

from .compare_particles import read_fields, read_grids

logging.basicConfig(level=logging.INFO)

STEPS = 300
LOWER_BOUNDARY = np.zeros(3)
NUMBER_OF_CELLS = np.array([192, 128, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])  # unit: meter
UPPER_BOUNDARY = NUMBER_OF_CELLS * CELL_SIZE + LOWER_BOUNDARY

GRID = Cartesian3DGrid(
    number_of_cells=NUMBER_OF_CELLS,
    lower_bound=LOWER_BOUNDARY,
    upper_bound=UPPER_BOUNDARY,
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
SOLVER = ElectromagneticSolver(grid=GRID, method="Yee", cfl=0.9)


PULSE_INIT = 15.0
LASER_DURATION = 5.0e-15
FOCAL_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
FOCAL_POSITION[1] = 4.62e-5
CENTROID_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
CENTROID_POSITION[1] = -0.5 * PULSE_INIT * LASER_DURATION * constants.c

LASERS = [
    GaussianLaser(
        wavelength=0.8e-6,
        waist=5.0e-6 / 1.17741,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        focal_position=FOCAL_POSITION,
        centroid_position=CENTROID_POSITION,
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    ),
    PlaneWaveLaser(
        wavelength=0.8e-6,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        centroid_position=CENTROID_POSITION,
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    ),
]


def basic_simulation():
    return Simulation(max_steps=STEPS, solver=SOLVER)


RUN_DIR = ""


def _inclusive_range(*args):
    """
    Implements range with inclusive endpoint, i.e., in the interval [,] instead of [,).
    """
    args = list(args)
    args[0 if len(args) == 1 else 1] += 1
    return range(*args)


def _make_inclusive(spec: slice):
    return slice(spec.start, spec.stop + 1 if spec.stop != -1 else None, spec.step)


def _indices(ts):
    # This function might need to change if the implementation details of
    # TimeStepSpec ever change.
    # It also relies on the picmi object and the pypicongpu object using
    # the same internal variable and storage layout.
    return sorted(reduce(set.union, (list(_inclusive_range(STEPS))[_make_inclusive(spec)] for spec in ts.specs), set()))


def setup_sim():
    sim = basic_simulation()
    for laser in LASERS:
        sim.add_laser(laser, None)
    sim.diagnostics = [Checkpoint(TimeStepSpec[::100])]
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(STEPS)
    return sim


SIM = None


class TestLasers(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM
        self.coordinates = np.transpose(
            np.meshgrid(
                *(
                    np.linspace(low, up, n, endpoint=False)
                    for low, up, n in zip(LOWER_BOUNDARY, UPPER_BOUNDARY, NUMBER_OF_CELLS)
                )
            ),
            (0, 2, 1, 3),
        )
        self.checkpoint_steps = _indices(
            self.sim.diagnostics[0].period.get_as_pypicongpu(self.sim.time_step_size, self.sim.max_steps)
        )

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_grid(self):
        np.testing.assert_allclose(
            read_grids(self.result_path / "simOutput" / "checkpoints" / "checkpoint_%T.bp5")["E"], self.coordinates
        )

    def test_total_E_field(self):
        for it in self.checkpoint_steps:
            expected = np.sum([laser.E(*self.coordinates, t=it * self.sim.time_step_size) for laser in LASERS], axis=0)
            fields = read_fields(self.result_path / "simOutput" / "checkpoints" / "checkpoint_%T.bp5", iteration=it)
            np.testing.assert_allclose(fields["E"], expected)
