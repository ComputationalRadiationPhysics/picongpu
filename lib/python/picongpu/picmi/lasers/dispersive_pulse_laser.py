"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

import numpy as np
from pydantic import model_validator

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .gaussian_laser import GaussianLaser


@default_converts_to(
    laser.DispersivePulseLaser,
    # PICMI's `duration` is the standard 1/e field width (tau), while PIConGPU's
    # `pulse_duration_si` (aliased as `duration`) is the 1 sigma of the intensity,
    # i.e. PULSE_DURATION = duration / 2 (#5739).
    #
    # As for the Gaussian laser, `pulse_init` is converted from the centroid-based
    # PICMI frame to the core's Huygens-surface frame at translation time.
    conversions={
        "duration": lambda self, *args, **kwargs: self._pulse_duration_sigma_si(),
        "pulse_init": lambda self, cell_size=None, domain_cells=None, *args, **kwargs: self._compute_pulse_init(
            cell_size, domain_cells
        ),
    },
)
class DispersivePulseLaser(GaussianLaser):
    """
    PICMI Dispersive Pulse Laser.

    Extends `GaussianLaser` with additional dispersion-specific parameters.

    Additional dispersive parameters (PIConGPU-specific):

    - picongpu_spectral_support : float, default=6.0
        Width of spectral support (dimensionless).
    - picongpu_sd_si : float, default=0.0
        Spatial dispersion coefficient [m*s].
    - picongpu_ad_si : float, default=0.0
        Angular dispersion coefficient [rad*s].
    - picongpu_gdd_si : float, default=0.0
        Group delay dispersion (GDD) [s^2].
    - picongpu_tod_si : float, default=0.0
        Third-order dispersion (TOD) [s^3].

    Notes
    -----
    Unlike the closed-form Gaussian/plane-wave profiles, the dispersive pulse's
    time-domain field is a *finite* discrete inverse Fourier transform of a
    closed-form frequency-domain field. It mirrors the C++ profile
    ``profiles::detail::DispersivePulseFunctorIncidentE`` in
    ``include/picongpu/fields/incidentField/profiles/DispersivePulse.hpp``.

    Because the transform is finite, the analytic field depends on the
    simulation's time step ``dt`` and on the (translation-derived) initialization
    duration ``pulse_init``. Both are threaded into :meth:`E` and
    :meth:`complex_amplitude` explicitly.
    """

    picongpu_spectral_support: float = 6.0
    picongpu_sd_si: float = 0.0
    picongpu_ad_si: float = 0.0
    picongpu_gdd_si: float = 0.0
    picongpu_tod_si: float = 0.0

    @model_validator(mode="wrap")
    @classmethod
    def _forbid_laguerre(cls, data, handler):
        if isinstance(data, dict):
            if data.get("picongpu_laguerre_modes", None) is not None:
                raise ValueError("DispersivePulseLaser does not support Laguerre modes.")
            if data.get("picongpu_laguerre_phases", None) is not None:
                raise ValueError("DispersivePulseLaser does not support Laguerre phases.")
        return handler(data)

    def _dispersive_init_time(self, pulse_init):
        """``INIT_TIME`` of the C++ profile, in seconds.

        ``INIT_TIME = PULSE_INIT * PULSE_DURATION`` with ``PULSE_INIT = pulse_init``
        (the translated, dimensionless initialization duration) and
        ``PULSE_DURATION = duration / 2`` (#5739).
        """
        return pulse_init * self._pulse_duration_sigma_si()

    def _expanded_wave_vector_x(self, d_omega, w0):
        """Port of ``DispersivePulseFunctorIncidentE::expandedWaveVectorX`` (SI)."""
        from picongpu.picmi.constants import c

        omega0 = self._Omega0()
        ad = self.picongpu_ad_si
        return (w0 / c) * (omega0 * ad * d_omega + ad * d_omega**2 - omega0 / 6.0 * ad**3 * d_omega**3)

    def _complex_amplitude_standard_conditions(self, x, y, z, t, dt=None, pulse_init=None):
        """Dispersive-pulse field in Gaussian-standard conditions.

        This is the frame used by ``GaussianLaser``: focus at the origin, propagation
        along ``+z`` and polarization along ``x`` at ``t = 0``. The C++ profile's
        laser-internal coordinates are ``(propagation, polarization, axis2) = (z, x, y)``,
        so the arguments are reordered below to match ``amp``/``phi``.

        ``dt`` is the simulation time step (finite inverse-DFT resolution) and
        ``pulse_init`` the translated initialization duration. Both are necessary;
        if ``pulse_init`` is omitted it is computed from the (grid-free) centroid
        convention (``origin`` at the coordinate origin).
        """
        from picongpu.picmi.constants import c

        if dt is None:
            raise ValueError(
                "DispersivePulseLaser.complex_amplitude() requires the simulation time step "
                "`dt` (in seconds): the pulse is a finite discrete inverse Fourier transform "
                "whose resolution is set by `dt`."
            )
        dt = float(dt)
        if not np.all(np.isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"DispersivePulseLaser needs a finite, positive `dt`. You gave {dt=}.")

        if pulse_init is None:
            pulse_init = self._compute_pulse_init()

        pulse_duration = self._pulse_duration_sigma_si()
        omega0 = self._Omega0()
        w0 = self.waist
        rayleigh_length = np.pi * w0**2 / self.wavelength
        init_time = self._dispersive_init_time(pulse_init)

        # Gaussian-standard coordinates -> C++ laser-internal coordinates
        # (propagation, polarization, axis2) = (z, x, y).
        prop = np.asarray(z, dtype=float)
        pol = np.asarray(x, dtype=float)
        axis2 = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)

        # Initialization window, expressed in the user-visible frame. In the C++
        # the field is zero unless 0 <= getTminusXoverC <= INIT_TIME; using the
        # #116 reference frame this is exactly
        #     -INIT_TIME/2 <= t - x/c <= +INIT_TIME/2     (x = propagation).
        window = np.abs(t - prop / c) <= 0.5 * init_time

        d_omega_k = 2.0 * np.pi / init_time
        # interpolation order of the DFT; int() truncates like C++ static_cast<int>
        n = int(0.5 * init_time / dt)
        sigma_omega = 1.0 / (np.sqrt(2.0) * pulse_duration)
        center_k = int(c * init_time / self.wavelength)
        min_omega_k = center_k - int(self.picongpu_spectral_support * sigma_omega / d_omega_k)
        k_min = max(min_omega_k, 1)
        k_max = min(2 * center_k - min_omega_k, n)

        def amp(omega):
            d_omega = omega - omega0
            waist = w0 * np.sqrt(1.0 + (prop / rayleigh_length) ** 2)
            alpha = self._expanded_wave_vector_x(d_omega, w0)
            center = self.picongpu_sd_si * d_omega - c * alpha * prop / (w0 * omega0)
            env_freq = -(d_omega**2) * pulse_duration**2
            env_pol = -((pol - center) ** 2) / waist**2
            env_axis2 = -(axis2**2) / waist**2
            mag = np.exp(env_freq + env_pol + env_axis2)
            # 3D normalization (Cartesian3DGrid only here)
            mag *= w0 / waist
            return mag * np.sqrt(np.pi) * 2.0 * pulse_duration * self.E0

        def phi(omega):
            d_omega = omega - omega0
            alpha = self._expanded_wave_vector_x(d_omega, w0)
            center = self.picongpu_sd_si * d_omega - c * alpha * prop / (w0 * omega0)
            phase = (
                omega * prop / c
                + 0.5 * self.picongpu_gdd_si * d_omega**2
                + self.picongpu_tod_si / 6.0 * d_omega**3
                + self.phi0
            )
            inverse_r = prop / (rayleigh_length**2 + prop**2)
            phase += ((pol - center) ** 2 + axis2**2) * omega * 0.5 * inverse_r / c
            phase -= alpha * pol / w0 + 0.25 * alpha**2 * prop / rayleigh_length
            # Gouy phase shift (3D)
            phase -= np.arctan(prop / rayleigh_length)
            return phase

        # shifted time at which the field is evaluated (C++ evaluationTime)
        evaluation_time = t

        shape = np.broadcast(prop, pol, axis2, t).shape
        e_t = np.zeros(shape, dtype=float)
        for k in range(k_min, k_max + 1):
            omega_k = k * d_omega_k
            e_t += amp(omega_k) * np.cos(phi(omega_k) - omega_k * evaluation_time)

        # standard normalization of the finite inverse DFT
        e_t /= dt * (2 * n + 1)
        return np.where(window, e_t, 0.0)

    def complex_amplitude(self, x, y, z, t=0.0, dt=None, pulse_init=None):
        x, y, z, t = self._to_standard_coordinates(x, y, z, t)
        return self._complex_amplitude_standard_conditions(x, y, z, t, dt=dt, pulse_init=pulse_init)

    def _polarization_vector_standard_conditions_at(self, x, y, z, t):
        # The C++ dispersive profile uses a constant linear polarization vector
        # (getAxis1() == polarization direction); no wavefront curvature is applied
        # to the vector itself. The time argument is included in the broadcast so
        # that time-array evaluations (e.g. envelope scans) keep the field shape.
        shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z), np.shape(t))
        return np.reshape(self.polarization_direction, (-1,) + (1,) * len(shape)) * np.ones((3,) + tuple(shape))

    def E(self, x, y, z, t=0.0, dt=None, pulse_init=None):
        value = np.real(self.complex_amplitude(x, y, z, t, dt=dt, pulse_init=pulse_init)).astype(float)
        return self.polarization_vector_at(x, y, z, t) * value[np.newaxis, ...]

    def Ex(self, x, y, z, t=0.0, dt=None, pulse_init=None):
        return self.E(x, y, z, t, dt=dt, pulse_init=pulse_init)[0]

    def Ey(self, x, y, z, t=0.0, dt=None, pulse_init=None):
        return self.E(x, y, z, t, dt=dt, pulse_init=pulse_init)[1]

    def Ez(self, x, y, z, t=0.0, dt=None, pulse_init=None):
        return self.E(x, y, z, t, dt=dt, pulse_init=pulse_init)[2]
