"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

PICMI frontend for the optical-imaging plugin (the upstream ``shadowgraphy``
plugin). ``OpticalImaging`` exposes the full configuration surface; the
``Shadowgraphy`` preset instantiates it with the canonical Tukey windows and
numerical-aperture band-pass mask.

The three mask functions (``positionWf``/``timeWf``/``maskFourier``) are
arbitrary user C++ in the plugin. Following the ``AnalyticDistribution``
pattern, the user supplies Python callables of the corresponding coordinates
that return sympy expressions; they are rendered into the C++ functions with
the ``PMAccPrinter``. The callables may reference the compile-time
``params::*`` constants (and ``sim.*`` quantities) by their C++ name, e.g. via
``sympy.Symbol("params::posWfSizeX")`` -- see the ``Shadowgraphy`` preset.
"""

from collections.abc import Callable
from typing import Any

import sympy
from picmistandard import PICMI_Diagnostic
from pydantic import ConfigDict, Field, model_validator

from picongpu.picmi._FieldFunctor import expression_from_callable
from picongpu.pypicongpu._field_functor import render as _render_expression
from picongpu.pypicongpu.output.optical_imaging import (
    COMPILE_TIME_FIELDS,
)
from picongpu.pypicongpu.output.optical_imaging import OpticalImaging as PyPIConGPUOpticalImaging


class _Params:
    """Attribute access to the compile-time ``params::*`` C++ symbols."""

    def __getattr__(self, name: str) -> sympy.Symbol:
        return sympy.Symbol(f"params::{name}")


def render_mask_function(function: Callable, variables: tuple[str, ...], context: str) -> str:
    """Render one user mask callable into PMacc C++.  See the module docstring.

    The callable receives one sympy ``Symbol`` per coordinate in ``variables``
    and must return a sympy expression. Any free symbol must either be one of
    the coordinates or a C++ name in the plugin's ``params::``/``sim.``
    namespace (rendered verbatim); anything else is rejected here rather than
    failing cryptically at device-compile time.
    """
    coordinates = {name: sympy.Symbol(name) for name in variables}
    expression = expression_from_callable(function, coordinates, {})
    if not isinstance(expression, sympy.Expr):
        expression = sympy.sympify(expression)
    for symbol in expression.free_symbols:
        name = str(symbol)
        if name in variables or name.startswith(("params::", "sim.", "math::", "pmacc::")):
            continue
        raise ValueError(
            f"{context} references the undefined symbol {name!r}. The mask functions may only depend on "
            f"their coordinates ({', '.join(variables)}) and the plugin's compile-time 'params::*' / 'sim.*' "
            "quantities (write those as sympy.Symbol('params::...'))."
        )
    return _render_expression(expression)


def _ramp(u, size, extent):
    """Sinusoidal-slope (Tukey-like) ramp from 0 to 1, the C++ window shape."""
    return sympy.Piecewise(
        ((1 - sympy.cos(sympy.pi * u / size)) / 2, u <= size),
        (sympy.cos(sympy.pi * (u - (extent - size)) / size) / 2 + sympy.Rational(1, 2), u > extent - size),
        (1, True),
    )


def _default_position_wf(i, j, plugin_num_x, plugin_num_y):
    params = _Params()
    return _ramp(i, params.posWfSizeX, plugin_num_x) * _ramp(j, params.posWfSizeY, plugin_num_y)


def _default_time_wf(t, sim_num_t):
    params = _Params()
    plugin_num_t = sim_num_t / params.tRes
    wf_size = params.tWfBuffer / params.tRes
    return sympy.Piecewise(
        ((1 - sympy.cos(sympy.pi * t / wf_size)) / 2, t < wf_size),
        (
            sympy.cos(sympy.pi * (t - (plugin_num_t - wf_size)) / wf_size) / 2 + sympy.Rational(1, 2),
            t > plugin_num_t - wf_size,
        ),
        (1, True),
    )


def _default_frequency_filter(omega):
    params = _Params()
    abs_omega = sympy.Abs(omega)
    d_omega_min = params.omegaMin - params.omegaWfMin
    d_omega_max = params.omegaWfMax - params.omegaMax
    return sympy.Piecewise(
        (1.0, (abs_omega >= params.omegaMin) & (abs_omega <= params.omegaMax)),
        (
            (1 - sympy.cos(sympy.pi * (abs_omega - params.omegaMin + d_omega_min) / d_omega_min)) / 2,
            (abs_omega > params.omegaWfMin) & (abs_omega < params.omegaMin),
        ),
        (
            (1 + sympy.cos(sympy.pi * (abs_omega - params.omegaMax) / d_omega_max)) / 2,
            (abs_omega > params.omegaMax) & (abs_omega < params.omegaWfMax),
        ),
        (0.0, True),
    )


def _default_numerical_aperture(kx, ky, omega):
    params = _Params()
    speed_of_light = sympy.Symbol("sim.si.getSpeedOfLight()")
    k_perp = sympy.sqrt(kx**2 + ky**2)
    k = sympy.Abs(omega / speed_of_light)
    offset_wf = params.centralOmega * params.numericalApertureWfSize / speed_of_light
    na_k = params.numericalAperture * k
    return sympy.Piecewise(
        (1.0, k_perp <= na_k),
        ((1 - sympy.cos(sympy.pi * (k_perp - na_k - offset_wf) / offset_wf)) / 2, k_perp <= na_k + offset_wf),
        (0.0, True),
    )


def _default_mask_fourier(kx, ky, omega):
    return _default_frequency_filter(omega) * _default_numerical_aperture(kx, ky, omega)


#: The user-facing compile-time values, mirroring ``shadowgraphy.param``. They
#: are shared by every ``OpticalImaging`` instance of a simulation (validated in
#: the backend), because the plugin's ``params::`` namespace is compile-time.
_COMPILE_TIME_FIELDS = COMPILE_TIME_FIELDS

_RUNTIME_FIELDS = (
    "start",
    "duration",
    "file",
    "ext",
    "slice_point",
    "focus_pos",
    "fourier_output",
    "final_output",
    "intermediate_output",
)


class OpticalImaging(PICMI_Diagnostic):
    """A general optical-imaging diagnostic (the upstream ``shadowgraphy`` plugin).

    The plugin time-integrates the Poynting vector in a fixed plane of the
    simulation and applies Fourier-domain masks; different applications
    (shadowgraphy, interferometry, ...) are combinations of the compile-time
    ``params`` values and the three mask functions. This class exposes the full
    surface; :class:`Shadowgraphy` is a ready-made preset.

    The plane is always the ``z = slice_point * extent`` plane and the probe
    pulse must propagate in ``z``. The plugin requires a **3D** simulation built
    with **FFTW3** and **openPMD** support; a 2D setup is rejected by the
    frontend.

    Parameters
    ----------
    duration: int
        Length of the time integration in simulation steps. Must be positive
        and is silently truncated to a multiple of ``t_res`` (the value the
        plugin actually uses).

    start: int
        Step at which the integration starts (default 0).

    file / ext: str
        Output file prefix (default ``"shadowgram"``) and openPMD backend
        (default ``"bp5"``).

    slice_point: float
        Position of the extraction plane as a ratio of the total ``z`` extent,
        ``0 <= slice_point < 1`` (default 0.5). Note that exactly ``1.0`` is
        outside the domain and rejected.

    focus_pos: float
        Focus position of the Fourier propagator relative to the slice point, in
        SI metres (default 0.0).

    fourier_output / final_output / intermediate_output: bool
        Optional openPMD outputs: the ``(x, y, omega)`` fields, the final
        shadowgram (requires a propagator run) and the ``(kx, ky, omega)``
        fields.

    t_res / x_res / y_res: int
        Time-integration and transverse resolutions (compile-time ``params``).

    numerical_aperture / numerical_aperture_wf_size: float
        Numerical aperture and width of its sinusoidal slope.

    central_lambda / d_lambda / d_lambda_wf: float
        Band-pass centre wavelength, plateau half-width and window-slope
        half-width, in SI metres.

    t_wf_buffer / pos_wf_size: int
        Time- and position-domain Tukey window buffer lengths.

    position_wf / time_wf / mask_fourier: callable
        The three mask functions. Each takes the coordinate symbols
        (``(i, j, pluginNumX, pluginNumY)``, ``(t, simNumT)`` and
        ``(kx, ky, omega)`` respectively) and returns a sympy expression. They
        may reference the compile-time ``params::*`` constants and ``sim.*``
        quantities by C++ name, e.g. ``sympy.Symbol("params::posWfSizeX")``.
        Unlike :class:`Shadowgraphy`, there is no default: a bare
        ``OpticalImaging`` requires all three.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # runtime (.cfg) options
    start: int = Field(0, ge=0)
    duration: int = Field(gt=0)
    file: str = "shadowgram"
    ext: str = "bp5"
    slice_point: float = Field(0.5, ge=0.0, lt=1.0)
    focus_pos: float = 0.0
    fourier_output: bool = False
    final_output: bool = False
    intermediate_output: bool = False

    # compile-time (params::) values
    t_res: int = Field(2, ge=1)
    x_res: int = Field(1, ge=1)
    y_res: int = Field(1, ge=1)
    numerical_aperture: float = Field(0.23, ge=0.0)
    central_lambda: float = Field(750e-9, gt=0.0)
    d_lambda: float = Field(20e-9, ge=0.0)
    t_wf_buffer: int = Field(32, ge=0)
    pos_wf_size: int = Field(12, ge=0)
    numerical_aperture_wf_size: float = Field(0.5, ge=0.0)
    d_lambda_wf: float = Field(20e-9, ge=0.0)

    # the three free mask functions
    position_wf: Callable[[Any, Any, Any, Any], sympy.Expr]
    time_wf: Callable[[Any, Any], sympy.Expr]
    mask_fourier: Callable[[Any, Any, Any], sympy.Expr]

    @model_validator(mode="after")
    def _round_duration_to_t_res(self):
        # the plugin silently truncates the duration to a multiple of tRes
        adjusted = (self.duration // self.t_res) * self.t_res
        if adjusted <= 0:
            raise ValueError(
                f"duration must be at least one t_res ({self.t_res}), but {self.duration} truncates to 0. "
                f"You gave duration={self.duration}, t_res={self.t_res}."
            )
        if adjusted != self.duration:
            self.duration = adjusted
        if self.d_lambda >= self.central_lambda:
            raise ValueError(
                "d_lambda must be smaller than central_lambda so that the band-pass plateau is non-empty. "
                f"You gave d_lambda={self.d_lambda}, central_lambda={self.central_lambda}."
            )
        return self

    def get_as_pypicongpu(
        self,
        time_step_size=None,
        num_steps=None,
        default_particle_shape=None,
    ) -> PyPIConGPUOpticalImaging:
        values = {name: getattr(self, name) for name in _RUNTIME_FIELDS + _COMPILE_TIME_FIELDS}
        return PyPIConGPUOpticalImaging(
            **values,
            position_wf_code=render_mask_function(
                self.position_wf, ("i", "j", "pluginNumX", "pluginNumY"), "position_wf"
            ),
            time_wf_code=render_mask_function(self.time_wf, ("t", "simNumT"), "time_wf"),
            mask_fourier_code=render_mask_function(self.mask_fourier, ("kx", "ky", "omega"), "mask_fourier"),
        )


class Shadowgraphy(OpticalImaging):
    """The shadowgraphy preset of :class:`OpticalImaging`.

    Fills in the canonical optical-imaging mask bundle -- Tukey windows in
    position and time, a band-pass in wavelength and a numerical-aperture mask
    -- and enables ``final_output`` so that a bare ``Shadowgraphy(...)`` call
    produces a shadowgram. All other parameters keep their
    :class:`OpticalImaging` defaults and may be overridden.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("position_wf", _default_position_wf)
        kwargs.setdefault("time_wf", _default_time_wf)
        kwargs.setdefault("mask_fourier", _default_mask_fourier)
        kwargs.setdefault("final_output", True)
        super().__init__(**kwargs)
