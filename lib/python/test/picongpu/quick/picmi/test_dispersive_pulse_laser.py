"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Off-GPU acceptance tests for ``DispersivePulseLaser`` (issue #117).

The dispersive-pulse time-domain field is a finite discrete inverse Fourier
transform of a closed-form frequency-domain field; the end-to-end comparison
against a compiled PIConGPU run needs a GPU and is deferred to CI.  Instead we
port the C++ profile functor ``profiles::detail::DispersivePulseFunctorIncidentE``
to a standalone Python oracle and assert that the analytic ``laser.E`` matches it
on a real e2e geometry, using the ``pulse_init`` produced by the translation
layer and the simulation time step ``dt`` -- and WITHOUT any ad-hoc timing shift.

The remaining tests are field-invariant checks mirroring
``test_gaussian_laser.py`` / ``test_plane_wave_laser.py``.
"""

from unittest import TestCase

import numpy as np
from scipy.constants import c as C

from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    DispersivePulseLaser,
    Simulation,
)
from picongpu.picmi.lasers import PolarizationType

# ---------------------------------------------------------------------------
# Port of the C++ core (SI units), mirroring test_laser_reference_frame.py.
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


class DispersiveOracle:
    """Port of ``DispersivePulseFunctorIncidentE`` (SI units, 3D)."""

    def __init__(self, laser, cell_size, domain_cells, dt, pulse_init):
        self.direction = np.asarray(laser.propagation_direction, dtype=float)
        self.polarization = np.asarray(laser.polarization_direction, dtype=float)
        self.axis2 = np.cross(self.direction, self.polarization)
        self.focus = np.asarray(laser.focal_position, dtype=float)
        self.origin = _get_origin(
            self.focus, self.direction, laser.picongpu_huygens_surface_positions, cell_size, domain_cells
        )
        self.dt = float(dt)
        self.wavelength = laser.wavelength
        self.amplitude = laser.E0
        self.pulse_duration = laser.duration / 2.0
        self.pulse_init = pulse_init
        self.omega0 = 2 * np.pi * C / self.wavelength
        self.w0 = laser.waist
        self.rayleigh_length = np.pi * self.w0**2 / self.wavelength
        self.spectral_support = laser.picongpu_spectral_support
        self.sd = laser.picongpu_sd_si
        self.ad = laser.picongpu_ad_si
        self.gdd = laser.picongpu_gdd_si
        self.tod = laser.picongpu_tod_si
        self.laser_phase = laser.phi0

    @property
    def init_time(self):
        return self.pulse_init * self.pulse_duration

    def expanded_wave_vector_x(self, d_omega):
        return (self.w0 / C) * (
            self.omega0 * self.ad * d_omega + self.ad * d_omega**2 - self.omega0 / 6.0 * self.ad**3 * d_omega**3
        )

    def amp(self, position, omega):
        d_omega = omega - self.omega0
        x, y, z = position
        waist = self.w0 * np.sqrt(1.0 + (x / self.rayleigh_length) ** 2)
        alpha = self.expanded_wave_vector_x(d_omega)
        center = self.sd * d_omega - C * alpha * x / (self.w0 * self.omega0)
        exponent = -(d_omega**2) * self.pulse_duration**2 - (y - center) ** 2 / waist**2 - z**2 / waist**2
        return np.exp(exponent) * (self.w0 / waist) * np.sqrt(np.pi) * 2.0 * self.pulse_duration * self.amplitude

    def phi(self, position, omega, phase_shift=0.0):
        d_omega = omega - self.omega0
        x, y, z = position
        alpha = self.expanded_wave_vector_x(d_omega)
        center = self.sd * d_omega - C * alpha * x / (self.w0 * self.omega0)
        phase = (
            omega * x / C + 0.5 * self.gdd * d_omega**2 + self.tod / 6.0 * d_omega**3 + phase_shift + self.laser_phase
        )
        inverse_r = x / (self.rayleigh_length**2 + x**2)
        phase += ((y - center) ** 2 + z**2) * omega * 0.5 * inverse_r / C
        phase -= alpha * y / self.w0 + 0.25 * alpha**2 * x / self.rayleigh_length
        phase -= np.arctan(x / self.rayleigh_length)
        return phase

    def scalar(self, position_si, time_si, phase_shift=0.0):
        position_si = np.asarray(position_si, dtype=float)
        shifted_time = time_si - ((position_si - self.origin) @ self.direction) / C
        if shifted_time < 0.0 or shifted_time > self.init_time:
            return 0.0
        shift_from_focus = position_si - self.focus
        internal = np.array(
            [
                shift_from_focus @ self.direction,
                shift_from_focus @ self.polarization,
                shift_from_focus @ self.axis2,
            ]
        )
        distance_origin_relative_to_focus = (self.origin - self.focus) @ self.direction
        mue = 0.5 * self.init_time
        time_delay = mue - distance_origin_relative_to_focus / C
        evaluation_time = time_si - time_delay

        d_omega_k = 2.0 * np.pi / self.init_time
        n = int(0.5 * self.init_time / self.dt)
        sigma_omega = 1.0 / (np.sqrt(2.0) * self.pulse_duration)
        center_k = int(C * self.init_time / self.wavelength)
        min_omega_k = center_k - int(self.spectral_support * sigma_omega / d_omega_k)
        k_min = max(min_omega_k, 1)
        k_max = min(2 * center_k - min_omega_k, n)

        e_t = 0.0
        for k in range(k_min, k_max + 1):
            omega_k = k * d_omega_k
            e_t += self.amp(internal, omega_k) * np.cos(
                self.phi(internal, omega_k, phase_shift) - omega_k * evaluation_time
            )
        return e_t / (self.dt * (2 * n + 1))

    def field(self, position_si, time_si):
        return self.polarization * self.scalar(position_si, time_si)


# ---------------------------------------------------------------------------
# Test geometry (mirrors the e2e test)
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
HUYGENS_SURFACE = [[16, -16], [16, -16], [16, -16]]

GRID = Cartesian3DGrid(
    number_of_cells=NUMBER_OF_CELLS.tolist(),
    lower_bound=np.zeros(3).tolist(),
    upper_bound=UPPER_BOUND.tolist(),
    lower_boundary_conditions=["open"] * 3,
    upper_boundary_conditions=["open"] * 3,
)
# Same solver/cfl as the e2e test, for the same time step `dt`.
TIME_STEP_SIZE = Simulation(max_steps=1, solver=ElectromagneticSolver(grid=GRID, method="Yee", cfl=0.9)).time_step_size

PROBE_Y_CELLS = np.array([30.0, 40.0, 50.0])
PROBE_CELLS = np.stack(
    [
        np.full_like(PROBE_Y_CELLS, FOCAL_POSITION[0] / CELL_SIZE[0]),
        PROBE_Y_CELLS,
        np.full_like(PROBE_Y_CELLS, FOCAL_POSITION[2] / CELL_SIZE[2]),
    ],
    axis=1,
)
# The pulse maximum is at the generation-surface origin when
# getTminusXoverC == INIT_TIME / 2; probe exactly there.
_ORIGIN = _get_origin(FOCAL_POSITION, [0.0, 1.0, 0.0], HUYGENS_SURFACE, CELL_SIZE, NUMBER_OF_CELLS)
PROBE_TIMES = (
    np.array([(cell * CELL_SIZE - _ORIGIN) @ np.array([0.0, 1.0, 0.0]) for cell in PROBE_CELLS]) / C
    + 0.5 * PULSE_INIT * LASER_DURATION_SIGMA
)


def _dispersive_laser(**kwargs):
    reference = dict(
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
    return DispersivePulseLaser(**(reference | kwargs))


def _translated_pulse_init(laser):
    return laser.get_as_pypicongpu(CELL_SIZE, NUMBER_OF_CELLS).pulse_init


class TestDispersivePulseLaserOracle(TestCase):
    """Analytic field matches the C++ finite inverse-DFT oracle, without shifts."""

    def _assert_matches_oracle(self, laser):
        pulse_init = _translated_pulse_init(laser)
        oracle = DispersiveOracle(laser, CELL_SIZE, NUMBER_OF_CELLS, TIME_STEP_SIZE, pulse_init)
        for cell, time in zip(PROBE_CELLS, PROBE_TIMES):
            position = cell * CELL_SIZE
            expected = oracle.field(position, time)
            found = laser.E(*position, t=time, dt=TIME_STEP_SIZE, pulse_init=pulse_init)
            self.assertGreater(np.abs(expected).max(), 1e-3 * laser.E0)
            np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-12 * laser.E0)

    def test_frontend_matches_core_oracle(self):
        self._assert_matches_oracle(_dispersive_laser())

    def test_frontend_matches_core_oracle_with_dispersion(self):
        self._assert_matches_oracle(
            _dispersive_laser(
                picongpu_sd_si=1e-20,
                picongpu_ad_si=1e-21,
                picongpu_gdd_si=1e-30,
                picongpu_tod_si=1e-46,
            )
        )

    def test_gate_zeros_outside_window(self):
        # Outside 0 <= getTminusXoverC <= INIT_TIME the C++ returns exactly zero.
        laser = _dispersive_laser()
        pulse_init = _translated_pulse_init(laser)
        cell = PROBE_CELLS[1]
        position = cell * CELL_SIZE
        before = PROBE_TIMES[1] - 2.0 * pulse_init * LASER_DURATION_SIGMA
        after = PROBE_TIMES[1] + 2.0 * pulse_init * LASER_DURATION_SIGMA
        np.testing.assert_array_equal(laser.E(*position, t=before, dt=TIME_STEP_SIZE, pulse_init=pulse_init), 0.0)
        np.testing.assert_array_equal(laser.E(*position, t=after, dt=TIME_STEP_SIZE, pulse_init=pulse_init), 0.0)
        self.assertNotEqual(
            float(np.abs(laser.E(*position, t=PROBE_TIMES[1], dt=TIME_STEP_SIZE, pulse_init=pulse_init)).max()),
            0.0,
        )

    def test_dt_is_required(self):
        laser = _dispersive_laser()
        with self.assertRaises(ValueError):
            laser.E(0.0, 0.0, 0.0, t=0.0)


class TestDispersivePulseLaserFieldComputation(TestCase):
    """Field-invariant checks of the analytic dispersive-pulse field."""

    def setUp(self):
        self.max_size = 50
        self.number_of_cells = 2 * self.max_size + 1
        self.grid = np.mgrid[: self.number_of_cells, : self.number_of_cells, : self.number_of_cells] - self.max_size
        self.reference_kwargs = dict(
            wavelength=4.0,
            waist=10.0,
            duration=10 / C,
            propagation_direction=[0, 1, 0],
            polarization_direction=[1, 0, 0],
            focal_position=[0, 0, 0],
            centroid_position=[0, 0, 0],
            a0=1.0,
        )
        # The finite inverse-DFT resolution: a realistic fraction of the carrier
        # period so that the spectral support is resolved.
        omega0 = 2 * np.pi * C / self.reference_kwargs["wavelength"]
        self.dt = (2 * np.pi / omega0) / 40.0
        # The C++ DFT parameter PULSE_INIT (dimensionless); passed explicitly here
        # so the field-invariant checks exercise the formula itself.
        self.pulse_init = 15.0

    def make_laser(self, **kwargs):
        return DispersivePulseLaser(**(self.reference_kwargs | kwargs))

    def test_on_axis_focus_amplitude_is_E0(self):
        # On-axis at the focus, around the time the pulse peak arrives there, the
        # envelope amplitude is the user-provided E0 (up to the finite-DFT
        # discretization of the spectrum, well below 1%).
        laser = self.make_laser()
        pulse_init = self.pulse_init
        period = 2 * np.pi / laser._Omega0()
        times = np.linspace(-period, period, 4001)
        values = laser.E(0.0, 0.0, 0.0, t=times, dt=self.dt, pulse_init=pulse_init)[0]
        peak = np.abs(values).max()
        self.assertGreater(peak, 0.98 * laser.E0)
        self.assertLess(peak, 1.01 * laser.E0)

    def test_E_has_component_first_layout(self):
        laser = self.make_laser()
        pulse_init = self.pulse_init
        e = laser.E(*self.grid, dt=self.dt, pulse_init=pulse_init)
        self.assertEqual(e.shape, (3,) + self.grid.shape[1:])
        np.testing.assert_allclose(
            e,
            np.stack(
                [
                    laser.Ex(*self.grid, dt=self.dt, pulse_init=pulse_init),
                    laser.Ey(*self.grid, dt=self.dt, pulse_init=pulse_init),
                    laser.Ez(*self.grid, dt=self.dt, pulse_init=pulse_init),
                ]
            ),
        )

    def test_polarization_layout(self):
        # The C++ dispersive profile applies a constant linear polarization vector.
        laser = self.make_laser()
        pulse_init = self.pulse_init
        grid = self.grid[:, :, self.max_size : self.max_size + 1, :]
        e = laser.E(*grid, dt=self.dt, pulse_init=pulse_init)
        # only the polarization component (x) is non-zero
        self.assertEqual(np.abs(e[1]).max(), 0.0)
        self.assertEqual(np.abs(e[2]).max(), 0.0)

    def test_duration_scaling(self):
        # PULSE_DURATION follows the Gaussian-family `duration / 2` convention, and
        # a longer `duration` gives a broader temporal envelope (same peak E0).
        for duration_sigma in (10 / C, 20 / C):
            laser = self.make_laser(duration=2 * duration_sigma)
            self.assertEqual(laser._pulse_duration_sigma_si(), duration_sigma)
            # The full width where |E| >= E0/e grows with the duration.
            pulse_init = self.pulse_init
            period = 2 * np.pi / laser._Omega0()
            times = np.linspace(-40 * duration_sigma, 40 * duration_sigma, 20001)
            values = np.abs(laser.E(0.0, 0.0, 0.0, t=times, dt=period / 200.0, pulse_init=pulse_init)[0])
            width = np.ptp(times[values >= values.max() / np.e])
            self.assertGreater(width, 0.5 * duration_sigma)
