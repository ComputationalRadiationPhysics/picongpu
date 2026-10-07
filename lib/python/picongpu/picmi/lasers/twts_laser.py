"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Alexander Debus
License: GPLv3+
"""

import logging
import math
from collections.abc import Sequence

from picmistandard import PICMI_Laser, resolve_once

import numpy as np
from pydantic import Field, computed_field, model_validator
from scipy.special import i0, jv

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser, PositiveFloat
from .polarization_type import PolarizationType

from .. import constants


def _blackman_nuttall_window(current_step, switch_start_step, switch_end_step, length):
    """Port of ``profiles::window::switchAt`` (TWTSPulse.def).

    The window parameters are time-step numbers. ``length < 0`` disables the
    window entirely.
    """
    a0, a1, a2, a3 = 0.3635819, 0.4891775, 0.1365995, 0.0106411
    if length < 0.0:
        return 1.0
    n_on = current_step - switch_start_step
    n_off = switch_end_step - current_step
    if (n_on < 0) or (n_off < 0):
        return 0.0
    if 0 <= n_on <= length:
        n = n_on
    elif (switch_end_step >= switch_start_step) and 0 <= n_off <= length:
        n = n_off
    else:
        return 1.0
    return (
        a0
        - a1 * math.cos(2.0 * math.pi * n / length / 2.0)
        + a2 * math.cos(4.0 * math.pi * n / length / 2.0)
        - a3 * math.cos(6.0 * math.pi * n / length / 2.0)
    )


@default_converts_to(
    laser.TWTSLaser,
    # PICMI's `duration` is the standard 1/e field width (tau), while PIConGPU's
    # `pulse_duration_si` (aliased as `duration`) is the 1 sigma of the intensity,
    # i.e. PULSE_DURATION = duration / 2 (#5739), exactly as for the Gaussian family.
    #
    # `focus_lateral_offset_si` is the core's "domain center + focus offset"
    # convention: the user only gives the absolute `focal_position`, and the grid
    # needed to convert it is known at translation time.
    conversions={
        "duration": lambda self, *args, **kwargs: self._pulse_duration_sigma_si(),
        "focus_lateral_offset_si": lambda self, cell_size=None, domain_cells=None, *args, **kwargs: (
            self._compute_focus_lateral_offset_si(cell_size, domain_cells)
        ),
    },
)
class TWTSLaser(PICMI_Laser, BaseLaser):
    """
    Specifies a Traveling-Wave Thomson Scattering (TWTS) laser

    Parameters
    ----------
    wavelength: float
        Central wavelength of the laser [m], must be > 0
    waist: float
        Spot size (1/e^2 radius) of the laser at focus [m], must be > 0
    duration: float
        Duration of the TWTS pulse [s], must be > 0. As for the Gaussian family
        this is the 1/e half-width of the electric-field envelope; the core's
        ``PULSE_DURATION`` is ``duration / 2`` (the 1 sigma of the intensity).
    laserIncidenceAngle: float
        Laser incidence angle [rad]
    polarizationAngle: float
        Linear laser polarization direction as rotation angle [rad]
    focal_position: list[float]
        3D coordinates of the laser focus [m]. This is the single independent
        focus-position input: its y-component sets the longitudinal focus and its
        z-component the lateral focus (relative to the domain center). The
        x-component is fixed to the domain center, mirroring the C++ TWTS origin.
    centroid_position: list[float]
        3D coordinates of the initial laser centroid [m]
    a0: float, optional
        Normalized vector potential at focus. Specify either a0 or E0.
    E0: float, optional
        Peak electric field amplitude [V/m]. Specify either a0 or E0.
    beta0: float, default 1.0
        Laser centroid speed normalized to speed of light, must be > 0
    windowStart: float, default 0.0
    windowEnd: float, default 0.0
    windowLength: float, default -1.0
        Blackman-Nuttall switch-on/off window length in time steps; a negative
        value (the default) disables the window.

    Notes
    -----
    Unlike the Gaussian/plane-wave/dispersive profiles, TWTS is not a
    ``BaseFunctorE``: its time reference is ``currentStep * dt - TDELAY`` with a
    user-facing ``time_offset_si`` (no Huygens-surface/pulse_init clock). The
    analytic field in :meth:`E` / :meth:`B` mirrors
    ``profiles::TWTSFunctorIncidentE/B`` and the closed-form
    ``templates::twtstight::EField/BField``. Because the C++ origin is the
    *domain center* laterally, :meth:`E` accepts the domain center as an explicit
    ``domain_center`` keyword (grid context that the laser object does not carry).
    """

    wavelength: PositiveFloat
    waist: PositiveFloat
    duration: PositiveFloat
    laserIncidenceAngle: float
    polarizationAngle: float
    focal_position: Sequence[float]
    centroid_position: Sequence[float]
    a0: float | None = None
    E0: float | None = None
    beta0: PositiveFloat = 1.0
    windowStart: float = 0.0
    windowEnd: float = 0.0
    # A negative length disables the Blackman-Nuttall switch-on/off window in the
    # core (``profiles::window::switchAt``); 0 is a *degenerate* zero-length window
    # in the core, so the disabled case is the sane default here.
    windowLength: float = -1.0

    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]]
    )
    picongpu_polarization_type: PolarizationType = PolarizationType.LINEAR

    @computed_field
    def pulse_init(self) -> float:
        return self._compute_twts_pulse_init()

    @computed_field
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @computed_field
    def phi0(self) -> float:
        # TWTS has no carrier phase input; the phase is always zero.
        return 0.0

    @computed_field
    def propagation_direction(self) -> list[float]:
        return [0.0, math.cos(self.laserIncidenceAngle), math.sin(self.laserIncidenceAngle)]

    @computed_field
    def polarization_direction(self) -> list[float]:
        # Rotation of the x-axis about the propagation direction by
        # `polarizationAngle` (Rodrigues). This is a unit vector for all angles,
        # unlike the previous expression which was only normalized at a few
        # special points.
        sin_phi = math.sin(self.laserIncidenceAngle)
        cos_phi = math.cos(self.laserIncidenceAngle)
        sin_angle = math.sin(self.polarizationAngle)
        cos_angle = math.cos(self.polarizationAngle)
        return [
            cos_angle,
            sin_phi * sin_angle,
            -cos_phi * sin_angle,
        ]

    @computed_field
    def laserIncidenceAnglePositive(self) -> bool:
        return self.laserIncidenceAngle > 0

    @computed_field
    def time_offset_si(self) -> float:
        return (self.focal_position[1] - self.centroid_position[1]) / (self.beta0 * constants.c)

    def _pulse_duration_sigma_si(self):
        """TWTS follows the Gaussian-family ``duration / 2`` convention (#5739)."""
        return self.duration / 2.0

    def _compute_twts_pulse_init(self):
        # TWTS always enters through the YMin face, so the beam travels along +y to
        # reach it. Keep the y-only expression (and its pre-existing semantics)
        # instead of the direction-generalized BaseLaser._compute_pulse_init();
        # with propagation_direction [0, cos(angle), sin(angle)] the generalized
        # form only reduces to this when laserIncidenceAngle == 0.
        pulse_init = (
            -2.0
            * self.centroid_position[1]
            / (self.propagation_direction[1] * constants.c)
            / self._pulse_duration_sigma_si()
        )
        if pulse_init < 3.0:
            logging.warning(
                "set centroid_position and propagation_direction indicate that laser "
                + "initalization might be too short.\n"
                + f"Details: {pulse_init=} < 3"
            )
        return pulse_init

    def _compute_focus_lateral_offset_si(self, cell_size=None, domain_cells=None):
        """Convert the absolute focus z-position into the core's domain-center offset.

        The C++ TWTS origin is the domain center laterally; the core parameter is
        ``FOCUS_Z_OFFSET_SI`` relative to that center. Without a grid (standalone
        translation) we fall back to the historical zero offset.
        """
        if cell_size is None or domain_cells is None:
            return 0.0
        domain_center = np.asarray(cell_size, dtype=float) * np.asarray(domain_cells, dtype=float) / 2.0
        return float(self.focal_position[2] - domain_center[2])

    def _validate_twts_properties(self):
        """Validation for the TWTS laser.

        TWTS is always placed on the YMin face, so it keeps its dedicated
        +y-entry validation (positive-y propagation, centroid_y <= 0) rather
        than the direction-generalized BaseLaser._validate_common_properties().
        """
        if not np.allclose(n := np.linalg.norm(self.polarization_direction), 1):
            raise ValueError(
                "The polarization direction vector must be normalized. "
                f"You gave {self.polarization_direction=} with norm {n}."
            )

        if not np.allclose(n := np.linalg.norm(self.propagation_direction), 1):
            raise ValueError(
                "The propagation direction vector must be normalized. "
                f"You gave {self.propagation_direction=} with norm {n}."
            )

        if self.propagation_direction[1] <= 0.0:
            raise ValueError(
                "Laser propagation parallel to the y-plane or pointing outside "
                "from the inside of the simulation box is not supported by this "
                f"laser in PICMI. You gave {self.propagation_direction=}."
            )

        if self.centroid_position[1] > 0:
            raise ValueError(
                "The laser maximum must be outside of the "
                "simulation box, otherwise it is impossible to correctly initialize"
                "it using a huygens surface in the box, centroid_y <= 0. "
                f"You gave {self.centroid_position=}."
            )

    @model_validator(mode="after")
    @resolve_once
    def _validate(self):
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, self.E0, self.a0)
        self._validate_twts_properties()
        return self

    def _twts_focus_si(self, domain_center=None):
        """TWTS coordinate origin (focus) in SI.

        The y- and z-components come from the user-visible ``focal_position``;
        the x-component is fixed to the domain center (the C++ origin is centered
        transversally in x). Without a grid the x-component is taken as given.
        """
        focus = np.asarray(self.focal_position, dtype=float).copy()
        if domain_center is not None:
            focus[0] = float(np.asarray(domain_center, dtype=float)[0])
        return focus

    def _twts_field_components(self, x, y, z, t=0.0, domain_center=None):
        """Closed-form TWTS E- and B-field, SI units.

        Direct port of the C++ ``templates::twtstight::TWTSTight<FieldE/FieldB>``
        chain (``TWTSTight.tpp`` + ``EField.tpp``/``BField.tpp``) evaluated in the
        simulation frame. ``t`` is the absolute simulation time; the C++ time
        reference ``currentStep * dt - tdelay`` is applied here via
        ``time_offset_si``. ``domain_center`` (SI, length 3) is the lateral grid
        context needed because the C++ places the origin at the domain center.
        """
        c = constants.c
        phi = float(self.laserIncidenceAngle)
        pol_angle = float(self.polarizationAngle)
        beta0 = float(self.beta0)
        focus = self._twts_focus_si(domain_center)
        time = np.asarray(t, dtype=float) - self.time_offset_si

        phi_positive = -1.0 if phi < 0.0 else 1.0
        abs_phi = abs(phi)
        sin_phi = math.sin(abs_phi)
        cos_phi = math.cos(abs_phi)
        tan_alpha = (1.0 - beta0 * cos_phi) / (beta0 * sin_phi)

        lambda0 = self.wavelength
        omega0 = 2.0 * math.pi * c / lambda0
        # factor 2 in tauG arises from the definition convention in the laser formula
        tau_g = self._pulse_duration_sigma_si() * 2.0
        w0 = self.waist
        k = 2.0 * math.pi / lambda0

        cot_phi = 1.0 / math.tan(abs_phi)
        sin_phi2 = sin_phi * sin_phi
        cos_phi2 = cos_phi * cos_phi
        sin_pol = math.sin(pol_angle)
        cos_pol = math.cos(pol_angle)
        sin2_phi = math.sin(2.0 * abs_phi)

        # reduced coordinates with wavelength-periodicity folding
        delta_t = lambda0 / c / (1.0 - beta0 * math.cos(phi))
        delta_y = beta0 * c * delta_t
        number_of_periods = np.floor(time / delta_t)
        time_mod = time - number_of_periods * delta_t
        y_mod = (np.asarray(y, dtype=float) - focus[1]) - number_of_periods * delta_y

        xr = phi_positive * (np.asarray(x, dtype=float) - focus[0])
        yr = y_mod
        zr = phi_positive * (np.asarray(z, dtype=float) - focus[2])
        tr = time_mod

        x2 = xr * xr
        tau_g2 = tau_g * tau_g
        psi0 = 2.0 / k
        w02 = w0 * w0
        beta02 = beta0 * beta0
        nu = (yr * cos_phi + zr * sin_phi) / c
        xi = (-zr * cos_phi + yr * sin_phi) * tan_alpha / c
        bessel_i0 = i0(k * k * sin_phi * w02 / 2.0)

        xm = -zr - 0.5j * (k * w02)
        rho_m = np.sqrt(x2 + xm**2)
        xm2 = xm * xm
        rho_m2 = rho_m * rho_m
        bessel_j0 = jv(0, k * sin_phi * rho_m)
        bessel_j1 = jv(1, k * sin_phi * rho_m)

        # Outside the pulse the (complex) envelope diverges exponentially; the C++
        # returns early in that case, so silence the transient overflow here and
        # rely on the gate mask below (identical result).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            zero_order = (beta0 * tau_g) / (
                math.sqrt(2.0)
                * np.exp(
                    beta02
                    * omega0
                    * (tr - nu - xi) ** 2
                    / (
                        beta02 * omega0 * tau_g2
                        - 2j * (beta02 * (nu - xi) * cot_phi * cot_phi)
                        + 2j * (beta0 * (2.0 * nu - xi) * cot_phi / sin_phi)
                        - 2j * (nu / sin_phi2)
                    )
                )
                * np.sqrt(
                    (
                        (beta02 * omega0 * tau_g2) / 2.0
                        - 1j * (beta02 * (nu - xi) * cot_phi * cot_phi)
                        + 1j * (beta0 * (2.0 * nu - xi) * cot_phi / sin_phi)
                        - 1j * (nu / sin_phi2)
                    )
                    / omega0
                )
            )
        phase = np.exp(1j * (omega0 * tr - k * yr * cos_phi))

        # Envelope gate: zero well outside the pulse (numSigmas = 6). The C++
        # evaluates this *before* the envelope (which overflows exponentially far
        # from the pulse), so mirror that by masking afterwards.
        gate = np.abs(yr - zr * tan_alpha - beta0 * c * tr) <= (6.0 * tau_g * c)

        e_x = phi_positive * np.real(
            0.25j
            * phase
            * zero_order
            * (
                k
                * rho_m
                * bessel_j0
                * (
                    (rho_m2 - x2 + xr * xm * cos_phi) * (sin_pol * sin_phi2)
                    + cos_pol * (rho_m2 + rho_m2 * cos_phi2 - x2 * sin_phi2 - xr * cos_phi * sin_phi2 * xm)
                )
                + bessel_j1
                * sin_phi
                * (
                    sin_pol
                    * (
                        -rho_m2
                        + 2.0 * x2
                        - 1j * rho_m2 * xm * (k * sin_phi)
                        + xr * cos_phi * (-2.0 * xm - 1j * rho_m2 * (k * sin_phi))
                    )
                    + cos_pol
                    * (
                        -rho_m2
                        + 2.0 * x2
                        + 1j * rho_m2 * xm * (k * sin_phi)
                        + xr * cos_phi * (2.0 * xm + 1j * rho_m2 * (k * sin_phi))
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
                bessel_j1 * (cos_pol * (xm - 2.0 * xr * cos_phi - xm * cos_phi2) + (1.0 + cos_phi2) * sin_pol * xm)
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
                    xr * (cos_pol + sin_pol) * sin_phi2 * xm
                    + cos_phi * (cos_pol * sin_phi2 * xm2 + sin_pol * (2.0 * rho_m2 - xm2 * sin_phi2))
                )
                + bessel_j1
                * sin_phi
                * (
                    cos_pol
                    * (
                        -4.0 * xr * xm
                        + 2.0 * cos_phi * (rho_m2 - 2.0 * xm2)
                        + 2j * rho_m2 * (xr - xm * cos_phi) * (k * sin_phi)
                    )
                    + sin_pol
                    * (
                        -4.0 * xr * xm
                        - 2.0 * cos_phi * (rho_m2 - 2.0 * xm2)
                        - 2j * rho_m2 * (k * xr * sin_phi)
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
                    cos_pol * (-rho_m2 + x2 + xr * cos_phi * xm) * sin_phi2
                    - sin_pol * (rho_m2 + rho_m2 * cos_phi2 - x2 * sin_phi2 + xr * cos_phi * sin_phi2 * xm)
                )
                + bessel_j1
                * (
                    cos_pol
                    * sin_phi
                    * (
                        rho_m2
                        - 2.0 * x2
                        + 1j * xm * rho_m2 * (k * sin_phi)
                        + xr * cos_phi * (-2.0 * xm - 1j * rho_m2 * (k * sin_phi))
                    )
                    + sin_pol
                    * (
                        (rho_m2 - 2.0 * x2) * sin_phi
                        + 1j * rho_m2 * (-xm + xr * cos_phi) * (k * sin_phi2)
                        + xr * sin2_phi * xm
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
                -(bessel_j1 * (cos_pol * (1.0 + cos_phi2) * xm + (xm + 2.0 * xr * cos_phi - xm * cos_phi2) * sin_pol))
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
                        xr * (2.0 * xm - 1j * (k * rho_m2 * sin_phi))
                        + cos_phi * (rho_m2 - 2.0 * xm2 - 1j * xm * (k * rho_m2 * sin_phi))
                    )
                    + cos_pol
                    * (
                        xr * (2.0 * xm + 1j * (k * rho_m2 * sin_phi))
                        + cos_phi * (-rho_m2 + 2.0 * xm2 + 1j * xm * (k * rho_m2 * sin_phi))
                    )
                )
                + k
                * rho_m
                * bessel_j0
                * (
                    xm * (-xr + xm * cos_phi) * (sin_pol * sin_phi2)
                    - cos_pol * (xr * sin_phi2 * xm + cos_phi * (-2.0 * rho_m2 + xm2 * sin_phi2))
                )
            )
            * psi0
            / (c * bessel_i0 * rho_m * rho_m2)
        )

        # The functor multiplies the amplitude-normalized profile by AMPLITUDE_SI.
        # ``np.where`` keeps the computation finite outside the gate (the raw
        # envelope overflows there, exactly as it would before the C++ early-out).
        e_field = np.where(gate, self.E0 * np.array([e_x, e_y, e_z]), 0.0)
        b_field = np.where(gate, self.E0 * np.array([b_x, b_y, b_z]), 0.0)
        return e_field, b_field

    def _window_factor(self, t, dt):
        if self.windowLength < 0.0:
            return 1.0
        if dt is None:
            raise ValueError(
                "TWTSLaser needs the simulation time step `dt` to evaluate its "
                "Blackman-Nuttall switch-on/off window (windowLength >= 0)."
            )
        current_step = np.asarray(t, dtype=float) / dt
        factor = np.vectorize(
            lambda step: _blackman_nuttall_window(float(step), self.windowStart, self.windowEnd, self.windowLength)
        )(current_step)
        return float(factor) if np.ndim(t) == 0 else factor

    def E(self, x, y, z, t=0.0, domain_center=None, dt=None):
        e_field, _ = self._twts_field_components(x, y, z, t=t, domain_center=domain_center)
        return e_field * self._window_factor(t, dt)

    def B(self, x, y, z, t=0.0, domain_center=None, dt=None):
        _, b_field = self._twts_field_components(x, y, z, t=t, domain_center=domain_center)
        return b_field * self._window_factor(t, dt)

    def Ex(self, x, y, z, t=0.0, domain_center=None, dt=None):
        return self.E(x, y, z, t=t, domain_center=domain_center, dt=dt)[0]

    def Ey(self, x, y, z, t=0.0, domain_center=None, dt=None):
        return self.E(x, y, z, t=t, domain_center=domain_center, dt=dt)[1]

    def Ez(self, x, y, z, t=0.0, domain_center=None, dt=None):
        return self.E(x, y, z, t=t, domain_center=domain_center, dt=dt)[2]

    def Bx(self, x, y, z, t=0.0, domain_center=None, dt=None):
        return self.B(x, y, z, t=t, domain_center=domain_center, dt=dt)[0]

    def By(self, x, y, z, t=0.0, domain_center=None, dt=None):
        return self.B(x, y, z, t=t, domain_center=domain_center, dt=dt)[1]

    def Bz(self, x, y, z, t=0.0, domain_center=None, dt=None):
        return self.B(x, y, z, t=t, domain_center=domain_center, dt=dt)[2]
