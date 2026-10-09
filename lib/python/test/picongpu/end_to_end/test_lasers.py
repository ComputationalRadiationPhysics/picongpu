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
    DispersivePulseLaser,
    ElectromagneticSolver,
    GaussianLaser,
    PlaneWaveLaser,
    Simulation,
    TWTSLaser,
    constants,
)
from picongpu.picmi import Species as Species
from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec
from picongpu.picmi.lasers import PolarizationType

from .arbitrary_parameters import gather_results
from .compare_particles import read_fields, read_grids

logging.basicConfig(level=logging.INFO)

# Both e2e classes run a 300-step 192x128x192 launch; the shared wait budget
# (TIMEOUT_COUNT in arbitrary_parameters.py x sleep_interval=5 s) must
# exceed one full 300-step run (~12 min on the reduced CI node); see the
# TIMEOUT_COUNT setting there.  300 steps is chosen so the laser peak
# (15*sigma_t ahead of the entry face, 11.25 um in domain length) crosses
# the YMin face exactly at the final checkpoint, keeping the +y test's
# strong-field comparison region non-empty at the last iteration.
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
# PIConGPU's PULSE_DURATION (1 sigma of the intensity) that the (already existing)
# simulation run used:
LASER_DURATION_SIGMA = 5.0e-15
# The PICMI-standard `duration` is the 1/e field width tau, i.e. twice the sigma
# (PULSE_DURATION = duration / 2); see GaussianLaser._pulse_duration_sigma_si (#5739).
LASER_DURATION = 2 * LASER_DURATION_SIGMA
DOMAIN_CENTER = NUMBER_OF_CELLS / 2 * CELL_SIZE
FOCAL_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
FOCAL_POSITION[1] = 4.62e-5
CENTROID_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
# pulse_init (a multiple of PULSE_DURATION) is derived from the centroid via the
# sigma, keep the same pulse_init=15 as in the existing run:
CENTROID_POSITION[1] = -0.5 * PULSE_INIT * LASER_DURATION_SIGMA * constants.c

# TWTS (issue #117). It is not a BaseFunctorE: its core time reference is
# `currentStep * dt - TDELAY` with TDELAY = time_offset_si (focal_y - centroid_y) /
# (beta0 c), and it enters through its default faces (YMin/ZMax here). Its focus
# sits at the domain center laterally, hence the analytic field needs the domain
# center as extra context. The parameters are chosen so the pulse is well inside
# the box (strong field) at the later checkpoints.
TWTS_DURATION_SIGMA = 2.0e-15
TWTS_DURATION = 2 * TWTS_DURATION_SIGMA
TWTS_FOCAL_POSITION = DOMAIN_CENTER.copy()
TWTS_FOCAL_POSITION[1] = 2.8e-6
TWTS_CENTROID_POSITION = DOMAIN_CENTER.copy()
TWTS_CENTROID_POSITION[1] = -8.0e-6

LASERS = [
    GaussianLaser(
        wavelength=0.8e-6,
        waist=5.0e-6 / 1.17741,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        focal_position=FOCAL_POSITION.tolist(),
        centroid_position=CENTROID_POSITION.tolist(),
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    ),
    PlaneWaveLaser(
        wavelength=0.8e-6,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        centroid_position=CENTROID_POSITION.tolist(),
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    ),
    # DispersivePulse (issue #117): same focus/centroid as the Gaussian laser, with
    # the dispersion terms left at their zero default (the finite inverse-DFT path
    # is exercised regardless).
    DispersivePulseLaser(
        wavelength=0.8e-6,
        waist=5.0e-6 / 1.17741,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        focal_position=FOCAL_POSITION.tolist(),
        centroid_position=CENTROID_POSITION.tolist(),
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    ),
    # TWTS (issue #117): uses its default injection faces (YMin + ZMax for
    # laserIncidenceAngle < 0); the tilted pulse front is visible in the field.
    TWTSLaser(
        wavelength=0.8e-6,
        waist=2.0e-6,
        duration=TWTS_DURATION,
        laserIncidenceAngle=-np.deg2rad(10.0),
        polarizationAngle=0.0,
        focal_position=TWTS_FOCAL_POSITION.tolist(),
        centroid_position=TWTS_CENTROID_POSITION.tolist(),
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
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
    sim.diagnostics = [Checkpoint(period=TimeStepSpec[::100])]
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = str(RUN_DIR)
    else:
        sim.step(STEPS)
    return sim


SIM = None


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
        # The generation surface sits at (index + 0.75) cells and the Huygens
        # box is applied on every face, so the total-field/scattered-field
        # correction lives in the cells adjacent to the surface (and is applied
        # twice where two faces meet, e.g. the XMin/ZMin corner of the oblique
        # multi-face pulse).  There the simulated field still carries the
        # discrete injection error, not the analytic incident field, so keep a
        # several-cell band clear of the surface on every side.
        margin = 8
        inner_min = int(np.max(mins) + margin)
        inner_max = int(np.min(maxs) - margin)
        # selector varies along mask-axis `axis` (mask layout: x, y, z)
        shape = [1, 1, 1]
        shape[axis] = -1
        selector = np.arange(domain_cells[axis], dtype=int).reshape(shape)
        mask &= (selector >= inner_min) & (selector <= inner_max)
    # openPMD data/mesh layout is (component, z, y, x)
    return np.transpose(mask, (2, 1, 0))


def _expected_E_field(coordinates, lasers, time, dt):
    """
    Sum of the analytic laser fields, evaluated at ``time`` (SI).

    The laser's ``pulse_init`` is converted to the core's Huygens-surface
    reference frame at translation time (see ``Simulation.get_as_pypicongpu``),
    so the analytic ``laser.E(..., t)`` and the simulation share one uniquely
    defined reference frame and no ad-hoc timing shift is needed.  The field
    exists only inside the Huygens box (masked separately).

    The ``DispersivePulseLaser`` field is a finite discrete inverse Fourier
    transform, so it additionally needs the global time step ``dt`` and the
    translated ``pulse_init``; both are supplied here (the translated value is
    exactly the one rendered into ``incidentField.param``).
    """
    total = None
    for laser in lasers:
        extra = {}
        if isinstance(laser, DispersivePulseLaser):
            extra["dt"] = dt
            extra["pulse_init"] = laser.get_as_pypicongpu(CELL_SIZE, NUMBER_OF_CELLS).pulse_init
        if isinstance(laser, TWTSLaser):
            # TWTS is not a BaseFunctorE: its time reference is already
            # `time_offset_si` (TDELAY) and its origin is the domain center
            # laterally, so the analytic field needs the domain-center context.
            # Its Blackman-Nuttall window also needs the global `dt`.
            extra["domain_center"] = DOMAIN_CENTER
            extra["dt"] = dt
        contribution = laser.E(*coordinates, t=time, **extra)
        total = contribution if total is None else total + contribution
    return total


def _strong_field_mask(field, threshold=0.05):
    """Cells where the analytic field magnitude is a significant fraction of its maximum."""
    magnitude = np.max(np.abs(field), axis=0)
    return magnitude > threshold * np.max(magnitude)


# The analytic reference is the continuum incident field, but the simulation
# advances the field with a discrete Yee solver whose numerical dispersion
# shifts the carrier phase by O(omega * dt) relative to the c-propagated
# analytic pulse.  This is a small error on the pulse body, but it dominates at
# carrier nodes (analytic amplitude ~ 0), where several laser contributions can
# also interfere destructively.  Restrict the comparison to the pulse body and
# use a tolerance that reflects the discrete-solver error instead of a tight
# amplitude match.
#
# A factor-of-two error (e.g. injecting the same pulse twice) changes the body
# amplitude by |E| >= threshold * max, which exceeds rtol * |E| + atol, so the
# check continues to reject double injection.
_LASER_FIELD_THRESHOLD = 0.3
_LASER_FIELD_RTOL = 0.25
_LASER_FIELD_ATOL_FRACTION = 0.1


class TestLasers(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
            self.sim = SIM
            gather_results(self.result_path)
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
        * only cells the laser has actually reached (strong field) are compared.

        No timing shift is applied: the reference-frame conversion lives in the
        translation layer, so ``laser.E(..., t)`` is already in the simulation's
        frame (see ``_expected_E_field``).
        """
        interior = _huygens_interior_mask(LASERS, CELL_SIZE, NUMBER_OF_CELLS)
        for it in self.checkpoint_steps:
            time = it * self.sim.time_step_size
            # ``read_fields``/``_huygens_interior_mask`` use the openPMD field
            # layout (component, z, y, x); transpose the analytic field into the
            # same layout so the mask and ``[:, region]`` indexing agree.  The
            # x/z cell count and cell size coincide here, so the old mismatch was
            # hidden for the x-z symmetric lasers.
            expected = np.transpose(
                _expected_E_field(self.coordinates, LASERS, time, self.sim.time_step_size), (0, 3, 2, 1)
            )
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

            region = interior & _strong_field_mask(expected, threshold=_LASER_FIELD_THRESHOLD)
            scale = np.abs(expected[:, region]).max()
            # The analytic reference is the continuum incident field, while the
            # simulation advances the field with the discrete Yee solver.  The
            # solver's numerical dispersion shifts the carrier phase relative to
            # the c-propagated analytic pulse (O(omega * dt) per step), which
            # dominates at carrier nodes and on the freshly injected ramp.  On
            # the pulse body this is a bounded amplitude/phase error, so compare
            # only the body and allow the discrete-solver tolerance.
            np.testing.assert_allclose(
                fields["E"][:, region],
                expected[:, region],
                rtol=_LASER_FIELD_RTOL,
                atol=_LASER_FIELD_ATOL_FRACTION * scale,
                err_msg=f"Simulated and analytic laser E field disagree at iteration {it}.",
            )


# ---------------------------------------------------------------------------
# One physical pulse injected from multiple Huygens faces
# (https://github.com/chillenzer-agents/picongpu/issues/180)
# ---------------------------------------------------------------------------
#
# An obliquely incident pulse whose propagation direction has non-zero x and z
# components crosses both the XMin and the ZMin face. The frontend injects the
# *same* profile under both face guards, so inside the box the field must equal
# the single analytic contribution of that one pulse -- not a sum of two
# independent pulses. This is the acceptance criterion agreed in the issue.

# 45 degrees in the x-z plane: with symmetric x/z components both the XMin and
# the ZMin face are crossed equally, so the two Huygens surfaces receive the same
# profile at the same retarded time (the "single pulse, not two" case).
MULTIFACE_PROPAGATION = np.array([1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)])
# The pulse propagates purely in the x-z plane (no y-component), so its beam
# axis is fixed in y. The box is only 128 cells (5.67 um) thick in y, unlike the
# +y setup above whose FOCAL_POSITION[1] is far out of plane. For the pulse to
# actually pass through the box volume, the axis must lie inside the box in y, so
# the focus is placed at the box center (DOMAIN_CENTER) on all axes.
MULTIFACE_FOCAL_POSITION = DOMAIN_CENTER.copy()
# The centroid sits 25 um up along -prop, just outside the box on both the x- and
# the z-entry sides (centroid_d*direction_d < 0 for both crossed entry faces).
# The peak then crosses both faces a few tens of steps in and the strong field is
# well inside the box by the it=100 checkpoint.
MULTIFACE_CENTROID_POSITION = MULTIFACE_FOCAL_POSITION - 25.0e-6 * MULTIFACE_PROPAGATION

MULTIFACE_LASER = GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=LASER_DURATION,
    propagation_direction=MULTIFACE_PROPAGATION.tolist(),
    polarization_direction=[0.0, 1.0, 0.0],
    focal_position=MULTIFACE_FOCAL_POSITION.tolist(),
    centroid_position=MULTIFACE_CENTROID_POSITION.tolist(),
    picongpu_polarization_type=PolarizationType.LINEAR,
    a0=8.0,
    phi0=0.0,
)

MULTIFACE_LASERS = [MULTIFACE_LASER]

MULTIFACE_SIM = None


def setup_multiface_sim():
    sim = basic_simulation()
    sim.add_laser(MULTIFACE_LASER, None)
    sim.diagnostics = [Checkpoint(period=TimeStepSpec[::100])]
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = str(RUN_DIR)
    else:
        sim.step(STEPS)
    return sim


class TestMultiFaceLaser(TestCase):
    """A single pulse injected through both its crossed faces (XMin and ZMin).

    The interior field is compared against the *single* analytic field of the one
    physical pulse, injected twice: the two Huygens contributions describe the
    same wavefront, so they must not add up to twice the field.
    """

    _result_path = None

    def setUp(self):
        global MULTIFACE_SIM
        assert MULTIFACE_LASER.picongpu_entry_faces is None
        assert MULTIFACE_LASER.entry_faces == ["XMin", "ZMin"]
        if MULTIFACE_SIM is None:
            MULTIFACE_SIM = setup_multiface_sim()
        self.sim = MULTIFACE_SIM
        gather_results(self.result_path)
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

    def test_multiface_incident_field_is_single_pulse(self):
        interior = _huygens_interior_mask(MULTIFACE_LASERS, CELL_SIZE, NUMBER_OF_CELLS)
        compared_any = False
        for it in self.checkpoint_steps:
            time = it * self.sim.time_step_size
            # one analytic contribution for the one physical pulse, transposed
            # into the openPMD (component, z, y, x) field layout (see TestLasers):
            expected = np.transpose(
                _expected_E_field(self.coordinates, MULTIFACE_LASERS, time, self.sim.time_step_size), (0, 3, 2, 1)
            )
            fields = read_fields(self.checkpoint_pattern, iteration=it)
            if it == self.checkpoint_steps[0]:
                self.assertLess(
                    np.abs(fields["E"]).max(),
                    1.0e-3 * np.abs(expected).max(),
                    f"Laser field present before the pulse has arrived (iteration {it}).",
                )
                continue
            region = interior & _strong_field_mask(expected, threshold=_LASER_FIELD_THRESHOLD)
            if not region.any():
                continue
            # Skip checkpoints at which the pulse has only just started to cross
            # the entry faces: there the strong-field region is the injection
            # ramp, where the discrete Huygens update has not yet settled into the
            # incident profile.  Once the analytic envelope maximum is inside the
            # Huygens box the comparison probes the injected pulse itself.
            if not interior.ravel()[np.argmax(np.max(np.abs(expected), axis=0))]:
                continue
            compared_any = True
            scale = np.abs(expected[:, region]).max()
            np.testing.assert_allclose(
                fields["E"][:, region],
                expected[:, region],
                rtol=_LASER_FIELD_RTOL,
                atol=_LASER_FIELD_ATOL_FRACTION * scale,
                err_msg=(
                    f"Multi-face (XMin+ZMin) injected field does not match the single analytic pulse "
                    f"at iteration {it}; the two Huygens faces must describe one wavefront, not two pulses."
                ),
            )
        self.assertTrue(compared_any, "No iteration had a strong enough field to compare.")
