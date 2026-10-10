"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Off-GPU acceptance test for the laser reference-frame fix (issue #116).

The end-to-end test that compares against a compiled PIConGPU run needs a GPU
and is deferred to CI.  Here we instead port the C++ incident-field evaluation
(``incidentField::detail::BaseFunctorE`` and the ``GaussianPulse``/``PlaneWave``
profile functors) to Python and assert that the analytic ``laser.E`` matches that
oracle -- with the ``pulse_init`` produced by the translation layer -- WITHOUT
any ad-hoc timing shift.  This pins the single, user-visible reference frame.
"""

import math
from unittest import TestCase

import numpy as np
from scipy.constants import c as C

from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GaussianLaser,
    PlaneWaveLaser,
    Simulation,
)
from picongpu.picmi.lasers import PolarizationType


# ---------------------------------------------------------------------------
# Port of the C++ core (SI units).  Cell indices are converted with cell_size.
# ---------------------------------------------------------------------------


def _get_origin(focus, direction, positions, cell_size, domain_cells):
    """Port of ``BaseFunctorE::getOrigin()`` (Functors.hpp)."""
    direction = np.asarray(direction, dtype=float)
    focus = np.asarray(focus, dtype=float)
    positions = np.asarray(positions, dtype=int)
    cell_size = np.asarray(cell_size, dtype=float)
    domain_cells = np.asarray(domain_cells, dtype=float)
    origin_p = -np.inf
    for axis in range(3):
        if abs(direction[axis]) <= np.finfo(float).eps:
            continue
        min_position = (positions[axis][0] + 0.75) * cell_size[axis]
        max_index = positions[axis][1] if positions[axis][1] > 0 else domain_cells[axis] + positions[axis][1]
        max_position = (max_index - 0.75) * cell_size[axis]
        axis_p = min(
            (min_position - focus[axis]) / direction[axis],
            (max_position - focus[axis]) / direction[axis],
        )
        origin_p = max(origin_p, axis_p)
    return focus + origin_p * direction


class _BaseEval:
    """Port of the coordinate/time transforms of ``BaseFunctorE``."""

    def __init__(self, direction, polarization, focus, origin, time_si):
        self.direction = np.asarray(direction, dtype=float)
        self.polarization = np.asarray(polarization, dtype=float)
        self.axis2 = np.cross(self.direction, self.polarization)
        self.focus = np.asarray(focus, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self.time_si = time_si

    def internal_coordinates(self, cell_index, cell_size):
        shift = np.asarray(cell_index, dtype=float) * cell_size - self.focus
        return np.array([shift @ self.direction, shift @ self.polarization, shift @ self.axis2])

    def t_minus_x_over_c(self, cell_index, cell_size):
        shift = np.asarray(cell_index, dtype=float) * cell_size - self.origin
        return self.time_si - (shift @ self.direction) / C


class GaussianOracle(_BaseEval):
    """Port of ``GaussianPulseFunctorIncidentE::getValue`` (m=0 default modes)."""

    def __init__(
        self,
        direction,
        polarization,
        focus,
        origin,
        time_si,
        wave_length,
        amplitude,
        pulse_duration,
        pulse_init,
        waist,
        laser_phase=0.0,
    ):
        super().__init__(direction, polarization, focus, origin, time_si)
        self.wave_length = wave_length
        self.amplitude = amplitude
        self.pulse_duration = pulse_duration
        self.w0 = waist
        self.laser_phase = laser_phase
        self.omega0 = 2 * np.pi * C / wave_length
        self.zr = np.pi * waist**2 / wave_length
        self.time_shift = 0.5 * pulse_init * pulse_duration

    def scalar(self, cell_index, cell_size):
        origin_relative = self.origin - self.focus
        distance = origin_relative @ self.direction
        time_delay = self.time_shift - distance / C
        internal = self.internal_coordinates(cell_index, cell_size)
        n = internal[0] / self.zr
        waist = self.w0 * np.sqrt(1 + n**2)
        inverse_r = internal[0] / (internal[0] ** 2 + self.zr**2)
        gouy = np.arctan(n)
        evaluation_time = (
            self.time_si
            - time_delay
            - internal[0] / C
            - internal[1] ** 2 * inverse_r / (2 * C)
            - internal[2] ** 2 * inverse_r / (2 * C)
        )
        amplitude_exponent = -(internal[1] ** 2 + internal[2] ** 2) / waist**2
        amplitude_abs = self.amplitude / np.sqrt(np.sqrt(1 + n**2)) / np.sqrt(np.sqrt(1 + n**2))
        phase = self.omega0 * evaluation_time + gouy + self.laser_phase
        e_spatial = amplitude_abs * np.exp(amplitude_exponent) * np.cos(phase)
        e_temporal = np.exp(-((evaluation_time / (2 * self.pulse_duration)) ** 2))
        return e_spatial * e_temporal

    def field(self, cell_index, cell_size):
        return self.polarization * self.scalar(cell_index, cell_size)


class PlaneWaveOracle(_BaseEval):
    """Port of ``PlaneWaveFunctorIncidentE::getLongitudinal`` (transversal = 1)."""

    def __init__(
        self,
        direction,
        polarization,
        focus,
        origin,
        time_si,
        wave_length,
        amplitude,
        pulse_duration,
        pulse_init,
        plateau,
        laser_phase=0.0,
    ):
        super().__init__(direction, polarization, focus, origin, time_si)
        self.wave_length = wave_length
        self.amplitude = amplitude
        self.pulse_duration = pulse_duration
        self.pulse_init = pulse_init
        self.plateau = plateau
        self.laser_phase = laser_phase
        self.omega0 = 2 * np.pi * C / wave_length

    def field(self, cell_index, cell_size):
        time = self.t_minus_x_over_c(cell_index, cell_size)
        envelope = self.amplitude
        mue = self.pulse_init * self.pulse_duration
        tau = self.pulse_duration * np.sqrt(2.0)
        start_down = mue + self.plateau
        correction = 0.0
        if time > start_down:
            envelope *= np.exp(-0.5 * ((time - start_down) / tau) ** 2)
            correction = (time - start_down) / (self.omega0 * tau * tau)
        elif time < mue:
            envelope *= np.exp(-0.5 * ((time - mue) / tau) ** 2)
            correction = (time - mue) / (self.omega0 * tau * tau)
        phase = self.omega0 * (time - mue) + self.laser_phase
        return self.polarization * (np.sin(phase) + np.cos(phase) * correction) * envelope


# ---------------------------------------------------------------------------
# Test geometry (mirrors the e2e test / a real PICMI setup)
# ---------------------------------------------------------------------------

NUMBER_OF_CELLS = np.array([192, 128, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])
UPPER_BOUND = NUMBER_OF_CELLS * CELL_SIZE
PULSE_INIT = 15.0
LASER_DURATION_SIGMA = 5.0e-15
LASER_DURATION = 2 * LASER_DURATION_SIGMA
FOCAL_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
FOCAL_POSITION[1] = 4.62e-5
CENTROID_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
CENTROID_POSITION[1] = -0.5 * PULSE_INIT * LASER_DURATION_SIGMA * C

GRID = Cartesian3DGrid(
    number_of_cells=NUMBER_OF_CELLS.tolist(),
    lower_bound=np.zeros(3).tolist(),
    upper_bound=UPPER_BOUND.tolist(),
    lower_boundary_conditions=["open"] * 3,
    upper_boundary_conditions=["open"] * 3,
)
HUYGENS_SURFACE = [[16, -16], [16, -16], [16, -16]]

# On-axis probe cells (x, z at the focus transverse position), safely inside the
# Huygens box, in cell indices.  We deliberately probe on the beam axis so the
# comparison is not affected by the frontend's wavefront-tilted polarization of
# the Gaussian laser (a separate modelling detail, not the reference frame).
PROBE_Y_CELLS = np.array([30.0, 40.0, 50.0])
PROBE_CELLS = np.stack(
    [
        np.full_like(PROBE_Y_CELLS, FOCAL_POSITION[0] / CELL_SIZE[0]),
        PROBE_Y_CELLS,
        np.full_like(PROBE_Y_CELLS, FOCAL_POSITION[2] / CELL_SIZE[2]),
    ],
    axis=1,
)
# Times (SI) at which the pulse maximum (the centroid at t=0) reaches each probe.
PROBE_TIMES = (PROBE_Y_CELLS * CELL_SIZE[1] - CENTROID_POSITION[1]) / C


def _gaussian_laser():
    return GaussianLaser(
        wavelength=0.8e-6,
        waist=5.0e-6 / 1.17741,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        focal_position=FOCAL_POSITION.tolist(),
        centroid_position=CENTROID_POSITION.tolist(),
        picongpu_polarization_type=PolarizationType.LINEAR,
        picongpu_huygens_surface_positions=HUYGENS_SURFACE,
        a0=8.0,
        phi0=0.0,
    )


def _plane_wave_laser(plateau_duration=0.0):
    return PlaneWaveLaser(
        wavelength=0.8e-6,
        duration=LASER_DURATION_SIGMA,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        centroid_position=CENTROID_POSITION.tolist(),
        picongpu_polarization_type=PolarizationType.LINEAR,
        picongpu_huygens_surface_positions=HUYGENS_SURFACE,
        picongpu_plateau_duration=plateau_duration,
        a0=8.0,
        phi0=0.0,
    )


def _translated_pulse_init(laser):
    """Run the laser through the translation layer to obtain its core pulse_init."""
    simulation = Simulation(
        max_steps=1,
        solver=ElectromagneticSolver(method="Yee", cfl=0.9, grid=GRID),
        lasers=[laser],
    )
    return simulation.get_as_pypicongpu().laser[0].pulse_init


class TestLaserReferenceFrame(TestCase):
    """The analytic laser fields match the C++ profile evaluation, without shifts."""

    def test_gaussian_frontend_matches_core_oracle(self):
        laser = _gaussian_laser()
        pulse_init = _translated_pulse_init(laser)
        direction = np.asarray(laser.propagation_direction, dtype=float)
        origin = _get_origin(
            laser.focal_position, direction, laser.picongpu_huygens_surface_positions, CELL_SIZE, NUMBER_OF_CELLS
        )
        oracle = GaussianOracle(
            direction=direction,
            polarization=laser.polarization_direction,
            focus=laser.focal_position,
            origin=origin,
            time_si=0.0,
            wave_length=laser.wavelength,
            amplitude=laser.E0,
            pulse_duration=laser._pulse_duration_sigma_si(),
            pulse_init=pulse_init,
            waist=laser.waist,
        )
        for cell, time in zip(PROBE_CELLS, PROBE_TIMES):
            oracle.time_si = time
            position = cell * CELL_SIZE
            expected = oracle.field(cell, CELL_SIZE)
            found = laser.E(*position, t=time)
            np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-6 * laser.E0)

    def _assert_plane_wave_matches_oracle(self, plateau_duration):
        # The plane-wave carrier can pass through a zero exactly at the envelope
        # peak, so compare over a full carrier period (plus any plateau) and only
        # require agreement where the field is actually significant.
        laser = _plane_wave_laser(plateau_duration=plateau_duration)
        pulse_init = _translated_pulse_init(laser)
        direction = np.asarray(laser.propagation_direction, dtype=float)
        origin = _get_origin(
            [0.0, 0.0, 0.0], direction, laser.picongpu_huygens_surface_positions, CELL_SIZE, NUMBER_OF_CELLS
        )
        oracle = PlaneWaveOracle(
            direction=direction,
            polarization=laser.polarization_direction,
            focus=[0.0, 0.0, 0.0],
            origin=origin,
            time_si=0.0,
            wave_length=laser.wavelength,
            amplitude=laser.E0,
            pulse_duration=laser._pulse_duration_sigma_si(),
            pulse_init=pulse_init,
            plateau=laser.picongpu_plateau_duration,
        )
        cell = PROBE_CELLS[1]
        time = PROBE_TIMES[1]
        period = 2 * np.pi / oracle.omega0
        span = max(0.5 * period, plateau_duration)
        times = time + np.linspace(-span, span, 101)
        expected = []
        found = []
        for t in times:
            oracle.time_si = t
            expected.append(oracle.field(cell, CELL_SIZE))
            found.append(laser.E(*(cell * CELL_SIZE), t=t))
        expected = np.array(expected)
        found = np.array(found)
        significant = np.abs(expected).max(axis=1) > 0.1 * laser.E0
        self.assertTrue(np.any(significant))
        np.testing.assert_allclose(found[significant], expected[significant], rtol=1e-6, atol=1e-6 * laser.E0)

    def test_plane_wave_frontend_matches_core_oracle(self):
        self._assert_plane_wave_matches_oracle(plateau_duration=0.0)

    def test_plane_wave_plateau_frontend_matches_core_oracle(self):
        # A non-zero plateau exercises the plateau branch of ``complex_amplitude``
        # (the ``time > start_down`` envelope/correction path) as well as the
        # plateau term in ``_compute_core_pulse_init`` -- the part of the frame
        # conversion where the centroid re-centring matters most.  The plateau
        # sits symmetrically around the pulse maximum at ``centroid_position`` at
        # t = 0 in the frontend and must match the C++ oracle.
        self._assert_plane_wave_matches_oracle(plateau_duration=4.0e-14)

    def test_pulse_init_uses_surface_origin_not_coordinate_origin(self):
        # The translated pulse_init must differ from the naive origin-at-zero
        # value: the reference-frame conversion is observable.
        laser = _gaussian_laser()
        pulse_init = _translated_pulse_init(laser)
        naive = (
            -2.0
            * float(np.dot(laser.centroid_position, laser.propagation_direction))
            / (C * laser._pulse_duration_sigma_si())
        )
        self.assertNotAlmostEqual(pulse_init, naive, places=6)

    def test_analytic_peak_is_at_centroid_at_t_zero(self):
        # The user-visible frame: the temporal envelope peaks at the centroid at
        # t=0 (self-consistent PICMI-standard formula) and its peak value equals
        # the complex amplitude evaluated there at t=0.
        laser = _gaussian_laser()
        centroid = np.asarray(laser.centroid_position)
        times = np.linspace(-5.0 * laser.duration, 5.0 * laser.duration, 20001)
        sampled = np.array([laser.envelope(*centroid, t=t) for t in times])
        self.assertAlmostEqual(times[np.argmax(np.abs(sampled))], 0.0, places=13)
        self.assertAlmostEqual(
            float(np.abs(sampled).max()), float(np.abs(laser.complex_amplitude(*centroid, t=0.0))), places=6
        )

    def test_oblique_geometry(self):
        # The frame conversion is direction-general: check a tilted propagation.
        # (Restrict the comparison to the transverse field-energy region; the
        # carrier must be sampled densely enough, exactly as the C++ oracle does.)
        direction = np.array([1.0, 3.0, 1.0]) / math.sqrt(11.0)
        polarization = np.cross([0.0, 0.0, 1.0], direction)
        polarization /= np.linalg.norm(polarization)
        focus = NUMBER_OF_CELLS / 2 * CELL_SIZE
        centroid = focus - 6.0e-5 * direction
        laser = GaussianLaser(
            wavelength=0.8e-6,
            waist=5.0e-6,
            duration=LASER_DURATION,
            propagation_direction=direction.tolist(),
            polarization_direction=polarization.tolist(),
            focal_position=focus.tolist(),
            centroid_position=centroid.tolist(),
            picongpu_polarization_type=PolarizationType.LINEAR,
            picongpu_huygens_surface_positions=HUYGENS_SURFACE,
            a0=8.0,
            phi0=0.0,
        )
        pulse_init = _translated_pulse_init(laser)
        origin = _get_origin(
            laser.focal_position, direction, laser.picongpu_huygens_surface_positions, CELL_SIZE, NUMBER_OF_CELLS
        )
        oracle = GaussianOracle(
            direction=direction,
            polarization=laser.polarization_direction,
            focus=laser.focal_position,
            origin=origin,
            time_si=0.0,
            wave_length=laser.wavelength,
            amplitude=laser.E0,
            pulse_duration=laser._pulse_duration_sigma_si(),
            pulse_init=pulse_init,
            waist=laser.waist,
        )
        # Probe at the focus when the pulse maximum (the centroid at t=0) arrives.
        axis_point = focus
        cell = axis_point / CELL_SIZE
        time = 6.0e-5 / C
        oracle.time_si = time
        expected = oracle.field(cell, CELL_SIZE)
        found = laser.E(*axis_point, t=time)
        self.assertGreater(np.abs(expected).max(), 1e-3 * laser.E0)
        np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-6 * laser.E0)
