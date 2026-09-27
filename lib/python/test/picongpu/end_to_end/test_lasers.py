"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
import os
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


# Set this (e.g. via $PICONGPU_LASER_TEST_RUN_DIR) to the directory of an
# existing PIConGPU run of *this exact* setup to short-circuit the heavy
# compile+run step and compare against the already-computed data instead; leave
# unset to compile and run the simulation (as done on the CI/HPC frontend).
_run_dir_env = os.environ.get("PICONGPU_LASER_TEST_RUN_DIR", "")
RUN_DIR = Path(_run_dir_env) if _run_dir_env else None
if RUN_DIR:
    logging.info(f"Reusing existing run output from {RUN_DIR}")


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
        sim.picongpu_get_runner().run_dir = str(RUN_DIR)
    else:
        sim.step(STEPS)
    return sim


SIM = None


def _huygens_origin(laser, cell_size, domain_cells):
    """
    Position of the laser center on the (Huygens) generation surface, in SI.

    Faithful port of ``incidentField::detail::BaseFunctorE::getOrigin()``:
    the origin is the intersection of the line through the focus along the
    (negative) propagation direction with the generation surface, choosing the
    point a laser encounters first.  The generation surface is displaced by
    0.75 cells inwards from the configured POSITION indices, matching the
    placement of the incident-field sources in the simulation.
    """
    direction = np.asarray(laser.propagation_direction, dtype=float)
    focus_attr = getattr(laser, "focal_position", None)
    if focus_attr is None:
        focus_attr = getattr(laser, "focus_pos", [0.0, 0.0, 0.0])
    focus = np.asarray(focus_attr, dtype=float)
    positions = np.asarray(laser.picongpu_huygens_surface_positions, dtype=int)
    origin_p = -np.inf
    for axis in range(3):
        if np.abs(direction[axis]) < 1e-30:
            continue
        min_pos = (positions[axis][0] + 0.75) * cell_size[axis]
        max_index = positions[axis][1] if positions[axis][1] > 0 else domain_cells[axis] + positions[axis][1]
        max_pos = (max_index - 0.75) * cell_size[axis]
        axis_p = min((min_pos - focus[axis]) / direction[axis], (max_pos - focus[axis]) / direction[axis])
        origin_p = max(origin_p, axis_p)
    return focus + origin_p * direction


def _huygens_interior_mask(lasers, cell_size, domain_cells):
    """
    Cells that are strictly inside the Huygens box (i.e. not in the PML/absorber
    layers between the domain boundaries and the generation surface).

    The analytic laser fields only describe the incident field fed in *through*
    the generation surface; inside the surrounding absorber layers the
    boundary conditions (not the laser model) determine the field, so those
    cells must be excluded from the comparison.
    """
    mask = np.ones(tuple(domain_cells), dtype=bool)
    for axis in range(3):
        # take the most restrictive positions across all lasers
        mins = np.array([laser.picongpu_huygens_surface_positions[axis][0] for laser in lasers])
        maxs = np.array(
            [
                (
                    laser.picongpu_huygens_surface_positions[axis][1]
                    if laser.picongpu_huygens_surface_positions[axis][1] > 0
                    else domain_cells[axis] + laser.picongpu_huygens_surface_positions[axis][1]
                )
                for laser in lasers
            ]
        )
        # inner points of the layer are at (index + 0.75); keep cells that are a
        # whole cell beyond it on either side
        inner_min = int(np.max(mins) + 1)
        inner_max = int(np.min(maxs) - 1)
        # selector varies along mask-axis `axis` (mask layout: x, y, z)
        shape = [1, 1, 1]
        shape[axis] = -1
        selector = np.arange(domain_cells[axis], dtype=int).reshape(shape)
        mask &= (selector >= inner_min) & (selector <= inner_max)
    # openPMD data/mesh layout is (component, z, y, x)
    return np.transpose(mask, (2, 1, 0))


def _expected_E_field(coordinates, lasers, time, cell_size, domain_cells):
    """
    Sum of the analytic laser fields, evaluated at ``time`` (SI).

    Models the following layers on top of the bare analytic formulas:

    * Huygens-surface timing: the frontend derives the laser's ``pulse_init``
      from the centroid under the assumption that the generation surface sits at
      the coordinate origin.  In reality the surface is displaced to the actual
      ``origin`` (see ``_huygens_origin``), which delays/advances the pulse by
      ``(origin . propagation_direction) / c``.  We add that shift so that the
      analytic prediction uses the same time reference as the simulation.
    * The field exists only inside the Huygens box (masked separately).
    """
    total = None
    for laser in lasers:
        origin = _huygens_origin(laser, cell_size, domain_cells)
        huygens_time_shift = np.dot(origin, laser.propagation_direction) / constants.c
        contribution = laser.E(*coordinates, t=time + huygens_time_shift)
        total = contribution if total is None else total + contribution
    return total


def _strong_field_mask(field, threshold=0.05):
    """Cells where the analytic field magnitude is a significant fraction of its maximum."""
    magnitude = np.max(np.abs(field), axis=0)
    return magnitude > threshold * np.max(magnitude)


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

    @property
    def checkpoint_pattern(self):
        return self.result_path / "simOutput" / "checkpoints" / "checkpoint_%T.bp5"

    def test_grid(self):
        np.testing.assert_allclose(read_grids(self.checkpoint_pattern)["E"], self.coordinates)

    def test_total_E_field(self):
        """
        The simulated E field equals the sum of the (analytic) laser fields,
        up to the distortions introduced by the numerical propagation layer.

        What is tested:

        * the FACET/laser implementation: the analytic formulas evaluate the very
          same incident-field profile that the simulation injects via the Huygens
          surface (the C++ ``GaussianPulse``/``PlaneWave`` functors);
        * the analytical formulas: amplitude (E0), temporal width (2*duration),
          Rayleigh length, Gouy phase, wavefront curvature;
        * the solver precision: numerical (Yee) dispersion/sampling distort the
          propagated pulse by a small amount, which bounds how tight the
          tolerances may be.

        Layers of distortion that are explicitly accounted for:

        * only the inside of the Huygens box is compared (the absorbing/PML
          layers near the boundaries are controlled by the boundary conditions,
          not by the laser model);
        * only cells the laser has actually reached (strong field) are compared;
        * the Huygens-surface timing shift (see ``_expected_E_field``).
        """
        interior = _huygens_interior_mask(LASERS, CELL_SIZE, NUMBER_OF_CELLS)
        for it in self.checkpoint_steps:
            time = it * self.sim.time_step_size
            expected = _expected_E_field(self.coordinates, LASERS, time, CELL_SIZE, NUMBER_OF_CELLS)
            fields = read_fields(self.checkpoint_pattern, iteration=it)

            if it == self.checkpoint_steps[0]:
                # Before the laser has entered the simulation volume there must
                # not be any field yet.
                self.assertLess(
                    np.abs(fields["E"]).max(),
                    1.0e-3 * np.abs(expected).max(),
                    f"Laser field present before the pulse has arrived (iteration {it}).",
                )
                continue

            region = interior & _strong_field_mask(expected)
            scale = np.abs(expected[:, region]).max()
            # rtol + atol such that the (dominant) field inside the reached region
            # agrees to within the numerical propagation error (~ a few % for the
            # Yee solver over the covered distance).
            np.testing.assert_allclose(
                fields["E"][:, region],
                expected[:, region],
                rtol=0.1,
                atol=0.1 * scale,
                err_msg=f"Simulated and analytic laser E field disagree at iteration {it}.",
            )
