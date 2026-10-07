"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Off-GPU acceptance tests for ``TWTSLaser`` (issue #117).

TWTS is a closed-form (but long) profile; the end-to-end comparison against a
compiled PIConGPU run needs a GPU and is deferred to CI.  Instead we port the
C++ ``templates::twtstight::TWTSTight<FieldE/FieldB>`` evaluation to a
standalone Python oracle and assert that the analytic ``laser.E``/``laser.B``
matches it on a real e2e geometry -- WITHOUT any ad-hoc timing shift.  The
remaining tests are field-invariant checks mirroring ``test_gaussian_laser.py``.
"""

from unittest import TestCase

import numpy as np
from scipy.constants import c as C
from scipy.special import i0, jv

from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    Simulation,
    TWTSLaser,
)
from picongpu.picmi.lasers import PolarizationType


# ---------------------------------------------------------------------------
# Standalone port of the C++ core (SI units), mirroring
# include/picongpu/fields/background/templates/twtstight.
# ---------------------------------------------------------------------------


def _twts_core(
    position,
    time,
    half_sim_size,
    cell_size,
    focus_y,
    wavelength,
    pulselength,
    w_x,
    phi,
    beta_0,
    focus_z_offset=0.0,
    polAngle=0.0,
    amplitude_si=1.0,
):
    """Port of ``TWTSTight<Field>::calcTWTSFieldX/Y/Z`` for E and B (SI, 3D).

    ``half_sim_size`` are the domain's half cell counts (C++) and the origin is
    the domain center laterally, ``focus_y``/``focus_z_offset`` longitudinally.
    Returns ``(E, B)`` as 3-vectors in SI.
    """
    c = C
    phi_positive = -1.0 if phi < 0.0 else 1.0
    abs_phi = abs(phi)
    sin_phi = np.sin(abs_phi)
    cos_phi = np.cos(abs_phi)
    tan_alpha = (1.0 - beta_0 * cos_phi) / (beta_0 * sin_phi)
    lambda0 = wavelength
    omega0 = 2.0 * np.pi * c / lambda0
    tau_g = pulselength * 2.0
    w0 = w_x
    k = 2.0 * np.pi / lambda0
    cot_phi = 1.0 / np.tan(abs_phi)
    sin_phi2 = sin_phi**2
    cos_phi2 = cos_phi**2
    sin_pol = np.sin(polAngle)
    cos_pol = np.cos(polAngle)
    sin2_phi = np.sin(2.0 * abs_phi)

    # getFieldPositions_SI: origin = domain center laterally + focus in y/z
    focus = np.array([half_sim_size[0], focus_y / cell_size[1], half_sim_size[2] + focus_z_offset / cell_size[2]])
    cell = np.asarray(position, dtype=float) / np.asarray(cell_size, dtype=float)
    pos = (cell - focus) * np.asarray(cell_size, dtype=float)

    delta_t = wavelength / c / (1.0 - beta_0 * np.cos(phi))
    delta_y = beta_0 * c * delta_t
    number_of_periods = np.floor(time / delta_t)
    time_mod = time - number_of_periods * delta_t
    y_mod = pos[1] - number_of_periods * delta_y

    x = phi_positive * pos[0]
    y = y_mod
    z = phi_positive * pos[2]
    t = time_mod

    x2 = x * x
    tau_g2 = tau_g * tau_g
    psi0 = 2.0 / k
    w02 = w0 * w0
    beta02 = beta_0 * beta_0
    nu = (y * cos_phi + z * sin_phi) / c
    xi = (-z * cos_phi + y * sin_phi) * tan_alpha / c
    bessel_i0 = i0(k * k * sin_phi * w02 / 2.0)
    xm = -z - 0.5j * (k * w02)
    rho_m = np.sqrt(x2 + xm**2)
    xm2 = xm * xm
    rho_m2 = rho_m * rho_m
    bessel_j0 = jv(0, k * sin_phi * rho_m)
    bessel_j1 = jv(1, k * sin_phi * rho_m)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        zero_order = (beta_0 * tau_g) / (
            np.sqrt(2.0)
            * np.exp(
                beta02
                * omega0
                * (t - nu - xi) ** 2
                / (
                    beta02 * omega0 * tau_g2
                    - 2j * (beta02 * (nu - xi) * cot_phi * cot_phi)
                    + 2j * (beta_0 * (2.0 * nu - xi) * cot_phi / sin_phi)
                    - 2j * (nu / sin_phi2)
                )
            )
            * np.sqrt(
                (
                    (beta02 * omega0 * tau_g2) / 2.0
                    - 1j * (beta02 * (nu - xi) * cot_phi * cot_phi)
                    + 1j * (beta_0 * (2.0 * nu - xi) * cot_phi / sin_phi)
                    - 1j * (nu / sin_phi2)
                )
                / omega0
            )
        )
    phase = np.exp(1j * (omega0 * t - k * y * cos_phi))

    e_x = phi_positive * np.real(
        0.25j
        * phase
        * zero_order
        * (
            k
            * rho_m
            * bessel_j0
            * (
                (rho_m2 - x2 + x * xm * cos_phi) * (sin_pol * sin_phi2)
                + cos_pol * (rho_m2 + rho_m2 * cos_phi2 - x2 * sin_phi2 - x * cos_phi * sin_phi2 * xm)
            )
            + bessel_j1
            * sin_phi
            * (
                sin_pol
                * (
                    -rho_m2
                    + 2.0 * x2
                    - 1j * rho_m2 * xm * (k * sin_phi)
                    + x * cos_phi * (-2.0 * xm - 1j * rho_m2 * (k * sin_phi))
                )
                + cos_pol
                * (
                    -rho_m2
                    + 2.0 * x2
                    + 1j * rho_m2 * xm * (k * sin_phi)
                    + x * cos_phi * (2.0 * xm + 1j * rho_m2 * (k * sin_phi))
                )
            )
        )
        * psi0
        / (rho_m * rho_m2 * bessel_i0)
    )
    e_y = np.real(
        phase
        * zero_order
        * (k * sin_phi)
        * (
            bessel_j1 * (cos_pol * (xm - 2.0 * x * cos_phi - xm * cos_phi2) + (1.0 + cos_phi2) * sin_pol * xm)
            + 1j * rho_m * bessel_j0 * ((cos_pol - sin_pol) * sin_phi2)
        )
        * psi0
        / (4.0 * bessel_i0 * rho_m)
    )
    e_z = phi_positive * np.real(
        0.125j
        * phase
        * zero_order
        * (
            2.0
            * k
            * rho_m
            * bessel_j0
            * (
                x * (cos_pol + sin_pol) * sin_phi2 * xm
                + cos_phi * (cos_pol * sin_phi2 * xm2 + sin_pol * (2.0 * rho_m2 - xm2 * sin_phi2))
            )
            + bessel_j1
            * sin_phi
            * (
                cos_pol
                * (
                    -4.0 * x * xm
                    + 2.0 * cos_phi * (rho_m2 - 2.0 * xm2)
                    + 2j * rho_m2 * (x - xm * cos_phi) * (k * sin_phi)
                )
                + sin_pol
                * (
                    -4.0 * x * xm
                    - 2.0 * cos_phi * (rho_m2 - 2.0 * xm2)
                    - 2j * rho_m2 * (k * x * sin_phi)
                    + 1j * rho_m2 * xm * (k * sin2_phi)
                )
            )
        )
        * psi0
        / (bessel_i0 * rho_m * rho_m2)
    )

    b_x = phi_positive * np.real(
        -0.25j
        * phase
        * zero_order
        * (
            k
            * rho_m
            * bessel_j0
            * (
                cos_pol * (-rho_m2 + x2 + x * cos_phi * xm) * sin_phi2
                - sin_pol * (rho_m2 + rho_m2 * cos_phi2 - x2 * sin_phi2 + x * cos_phi * sin_phi2 * xm)
            )
            + bessel_j1
            * (
                cos_pol
                * sin_phi
                * (
                    rho_m2
                    - 2.0 * x2
                    + 1j * xm * rho_m2 * (k * sin_phi)
                    + x * cos_phi * (-2.0 * xm - 1j * rho_m2 * (k * sin_phi))
                )
                + sin_pol
                * (
                    (rho_m2 - 2.0 * x2) * sin_phi
                    + 1j * rho_m2 * (-xm + x * cos_phi) * (k * sin_phi2)
                    + x * sin2_phi * xm
                )
            )
        )
        * psi0
        / (c * bessel_i0 * rho_m * rho_m2)
    )
    b_y = np.real(
        phase
        * zero_order
        * (k * sin_phi)
        * (
            -(bessel_j1 * (cos_pol * (1.0 + cos_phi2) * xm + (xm + 2.0 * x * cos_phi - xm * cos_phi2) * sin_pol))
            + 1j * rho_m * bessel_j0 * (cos_pol - sin_pol) * sin_phi2
        )
        * psi0
        / (4.0 * c * bessel_i0 * rho_m)
    )
    b_z = phi_positive * np.real(
        -0.25j
        * phase
        * zero_order
        * (
            bessel_j1
            * sin_phi
            * (
                sin_pol
                * (
                    x * (2.0 * xm - 1j * (k * rho_m2 * sin_phi))
                    + cos_phi * (rho_m2 - 2.0 * xm2 - 1j * xm * (k * rho_m2 * sin_phi))
                )
                + cos_pol
                * (
                    x * (2.0 * xm + 1j * (k * rho_m2 * sin_phi))
                    + cos_phi * (-rho_m2 + 2.0 * xm2 + 1j * xm * (k * rho_m2 * sin_phi))
                )
            )
            + k
            * rho_m
            * bessel_j0
            * (
                xm * (-x + xm * cos_phi) * (sin_pol * sin_phi2)
                - cos_pol * (x * sin_phi2 * xm + cos_phi * (-2.0 * rho_m2 + xm2 * sin_phi2))
            )
        )
        * psi0
        / (c * bessel_i0 * rho_m * rho_m2)
    )

    gate = np.abs(y - z * tan_alpha - beta_0 * c * t) <= (6.0 * tau_g * c)
    if not gate:
        return np.zeros(3), np.zeros(3)
    amplitude = amplitude_si
    return amplitude * np.array([e_x, e_y, e_z]), amplitude * np.array([b_x, b_y, b_z])


class TWTSOracle:
    """C++-mirror TWTS evaluator on a fixed grid."""

    def __init__(self, laser, cell_size, domain_cells, domain_center):
        self.laser = laser
        self.cell_size = np.asarray(cell_size, dtype=float)
        self.half_sim_size = np.asarray(domain_cells, dtype=float) / 2.0
        self.domain_center = np.asarray(domain_center, dtype=float)
        self.focus_y = laser.focal_position[1]
        self.focus_z_offset = laser.focal_position[2] - self.domain_center[2]

    def field(self, position, time):
        # The core time reference is `currentStep * dt - TDELAY` (TWTS is not a
        # BaseFunctorE; TDELAY = the frontend's time_offset_si).
        return _twts_core(
            position,
            time - self.laser.time_offset_si,
            self.half_sim_size,
            self.cell_size,
            self.focus_y,
            self.laser.wavelength,
            self.laser._pulse_duration_sigma_si(),
            self.laser.waist,
            self.laser.laserIncidenceAngle,
            self.laser.beta0,
            self.focus_z_offset,
            self.laser.polarizationAngle,
            amplitude_si=self.laser.E0,
        )


# ---------------------------------------------------------------------------
# Test geometry (mirrors the e2e test)
# ---------------------------------------------------------------------------

NUMBER_OF_CELLS = np.array([192, 128, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])
UPPER_BOUND = NUMBER_OF_CELLS * CELL_SIZE
DOMAIN_CENTER = UPPER_BOUND / 2.0
HUYGENS_SURFACE = [[16, -16], [16, -16], [16, -16]]
TIME_STEP_SIZE = None  # set below

GRID = Cartesian3DGrid(
    number_of_cells=NUMBER_OF_CELLS.tolist(),
    lower_bound=np.zeros(3).tolist(),
    upper_bound=UPPER_BOUND.tolist(),
    lower_boundary_conditions=["open"] * 3,
    upper_boundary_conditions=["open"] * 3,
)
TIME_STEP_SIZE = Simulation(max_steps=1, solver=ElectromagneticSolver(grid=GRID, method="Yee", cfl=0.9)).time_step_size

# Probe cells on the tilted pulse front at the time step used by the oracle test,
# chosen where |E| is a significant fraction of E0 (not in the exponentially small
# far tail, where the comparison would be trivial).
PROBE_STEP = 200
PROBE_CELLS = np.array(
    [
        [96.0, 0.0, 10.0],
        [96.0, 2.0, 7.0],
        [96.0, 4.0, 4.0],
    ]
)


def _twts_laser(**kwargs):
    reference = dict(
        wavelength=0.8e-6,
        waist=2.0e-6,
        duration=4.0e-15,
        laserIncidenceAngle=np.deg2rad(10.0),
        polarizationAngle=0.0,
        focal_position=[DOMAIN_CENTER[0], 2.8e-6, DOMAIN_CENTER[2]],
        centroid_position=[DOMAIN_CENTER[0], -8.0e-6, DOMAIN_CENTER[2]],
        picongpu_polarization_type=PolarizationType.LINEAR,
        picongpu_huygens_surface_positions=HUYGENS_SURFACE,
        a0=8.0,
    )
    return TWTSLaser(**(reference | kwargs))


class TestTWTSLaserOracle(TestCase):
    """Analytic field matches the C++ closed-form oracle, without shifts."""

    def _assert_matches_oracle(self, laser, time_step, probe_cells=None):
        oracle = TWTSOracle(laser, CELL_SIZE, NUMBER_OF_CELLS, DOMAIN_CENTER)
        for cell in PROBE_CELLS if probe_cells is None else probe_cells:
            position = cell * CELL_SIZE
            time = time_step * TIME_STEP_SIZE
            expected_e, expected_b = oracle.field(position, time)
            found_e = laser.E(*position, t=time, domain_center=DOMAIN_CENTER)
            found_b = laser.B(*position, t=time, domain_center=DOMAIN_CENTER)
            # The probe cells sit in the strong-field part of the pulse, so the
            # match is non-trivial.
            self.assertGreater(np.abs(expected_e).max(), 0.1 * laser.E0)
            np.testing.assert_allclose(found_e, expected_e, rtol=1e-12, atol=1e-12 * laser.E0)
            np.testing.assert_allclose(found_b, expected_b, rtol=1e-12, atol=1e-12 * laser.E0 / C)

    def test_frontend_matches_core_oracle(self):
        laser = _twts_laser()
        self._assert_matches_oracle(laser, 200)

    def test_frontend_matches_core_oracle_negative_angle(self):
        laser = _twts_laser(laserIncidenceAngle=-np.deg2rad(10.0))
        # A negative angle mirrors the tilted pulse front in z, so probe at zMax.
        self._assert_matches_oracle(
            laser, 200, probe_cells=np.array([[96.0, 0.0, 120.0], [96.0, 2.0, 123.0], [96.0, 3.0, 124.0]])
        )

    def test_frontend_matches_core_oracle_polarization(self):
        laser = _twts_laser(polarizationAngle=np.deg2rad(30.0))
        self._assert_matches_oracle(laser, 200)

    def test_gate_zeros_outside_envelope(self):
        laser = _twts_laser()
        oracle = TWTSOracle(laser, CELL_SIZE, NUMBER_OF_CELLS, DOMAIN_CENTER)
        # A point far outside the (numerically unstable) envelope gate must be
        # exactly zero, as in the C++ early return.
        position = PROBE_CELLS[0] * CELL_SIZE
        for time in (0.0, 1.0e-12):
            expected_e, expected_b = oracle.field(position, time)
            found_e = laser.E(*position, t=time, domain_center=DOMAIN_CENTER)
            if np.all(expected_e == 0.0):
                np.testing.assert_array_equal(found_e, 0.0)


class TestTWTSLaserFieldComputation(TestCase):
    """Field-invariant checks of the analytic TWTS field."""

    def setUp(self):
        self.reference_kwargs = dict(
            wavelength=0.8e-6,
            waist=2.0e-6,
            duration=4.0e-15,
            laserIncidenceAngle=np.deg2rad(10.0),
            polarizationAngle=0.0,
            focal_position=[0.0, 0.0, 0.0],
            centroid_position=[0.0, 0.0, 0.0],
            a0=1.0,
        )

    def make_laser(self, **kwargs):
        return TWTSLaser(**(self.reference_kwargs | kwargs))

    def test_E_has_component_first_layout(self):
        laser = self.make_laser()
        found = laser.E(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), t=0.0)
        self.assertEqual(found.shape, (3, 2, 2, 2))
        np.testing.assert_allclose(
            found,
            np.stack(
                [
                    laser.Ex(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), t=0.0),
                    laser.Ey(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), t=0.0),
                    laser.Ez(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), t=0.0),
                ]
            ),
        )

    def test_duration_conversion(self):
        # TWTS follows the Gaussian-family `duration / 2` convention (#5739).
        laser = self.make_laser()
        self.assertEqual(laser._pulse_duration_sigma_si(), laser.duration / 2.0)
        self.assertEqual(laser.get_as_pypicongpu().pulse_duration_si, laser.duration / 2.0)

    def test_focus_offset_translated_relative_to_domain_center(self):
        # The single user-visible focus z-position is converted into the core's
        # "domain center + focus offset" convention at translation time.
        laser = self.make_laser(focal_position=[0.0, 1.0e-6, 2.0e-6])
        translated = laser.get_as_pypicongpu(CELL_SIZE, NUMBER_OF_CELLS)
        self.assertAlmostEqual(translated.focus_lateral_offset_si, 2.0e-6 - DOMAIN_CENTER[2])
        # without a grid the historical zero offset is used
        self.assertEqual(laser.get_as_pypicongpu().focus_lateral_offset_si, 0.0)

    def test_gate_zeros_field_far_from_pulse(self):
        # Far ahead/behind the tilted pulse front the envelope is gated to zero.
        laser = self.make_laser()
        e = laser.E(0.0, 0.0, 0.0, t=1.0e-12)
        np.testing.assert_array_equal(e, 0.0)

    def test_phi_positive_sign_flip(self):
        # A 180-degree rotation about the y-axis flips Ex and Ez for negative
        # incidence angles, with the transverse mirror x -> -x, z -> -z.
        positive = self.make_laser(laserIncidenceAngle=np.deg2rad(10.0))
        negative = self.make_laser(laserIncidenceAngle=-np.deg2rad(10.0))
        x, y, z = 1.0e-7, 2.0e-6, 1.0e-6
        t = 2.3e-14
        e_pos = positive.E(x, y, z, t=t)
        e_neg = negative.E(-x, y, -z, t=t)
        for component, sign in ((0, -1.0), (1, 1.0), (2, -1.0)):
            scale = max(np.abs(e_pos[component]), abs(e_neg[component]), 1e-30)
            self.assertLess(np.abs(e_pos[component] - sign * e_neg[component]) / scale, 1e-9)
