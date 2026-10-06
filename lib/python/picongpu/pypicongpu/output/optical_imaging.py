"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class OpticalImaging(BaseModel):
    """Rendered configuration of (one instance of) the shadowgraphy/optical-imaging plugin.

    The plugin time-integrates the Poynting vector in a fixed plane and applies
    Fourier-domain masks; the ``Shadowgraphy`` PICMI preset translates into an
    object of this class. Every instance of this model renders into one entry of
    the plugin's multi-instance option list (``--shadowgraphy.*``) and shares the
    compile-time ``params::`` constants rendered into ``shadowgraphy.param``.
    """

    # runtime (.cfg) options, one entry of the plugin's multi-instance list each
    start: int = Field(0, ge=0, description="Step at which the time integration starts.")
    duration: int = Field(gt=0, description="Integration duration in steps, already truncated to a multiple of t_res.")
    file: str = Field("shadowgram", description="Output file prefix.")
    ext: str = Field("bp5", description="openPMD backend extension.")
    slice_point: float = Field(0.5, ge=0.0, lt=1.0, description="Slice position as a ratio of the z extent, [0, 1).")
    focus_pos: float = Field(
        0.0, description="Focus position of the Fourier propagator relative to the slice point, in SI metres."
    )
    fourier_output: bool = Field(False, description="Also dump the (x, y, omega) Fourier fields.")
    final_output: bool = Field(False, description="Run the propagator and write the final shadowgram.")
    intermediate_output: bool = Field(False, description="Dump the (kx, ky, omega) Fourier fields.")

    # compile-time (params::) constants, rendered into shadowgraphy.param
    t_res: int = Field(2, ge=1, description="Time-integration resolution.")
    x_res: int = Field(1, ge=1, description="Transverse resolution in x.")
    y_res: int = Field(1, ge=1, description="Transverse resolution in y.")
    numerical_aperture: float = Field(0.23, ge=0.0)
    central_lambda: float = Field(750e-9, gt=0.0)
    d_lambda: float = Field(20e-9, ge=0.0)
    t_wf_buffer: int = Field(32, ge=0)
    pos_wf_size: int = Field(12, ge=0)
    numerical_aperture_wf_size: float = Field(0.5, ge=0.0)
    d_lambda_wf: float = Field(20e-9, ge=0.0)

    # the three free mask functions, already rendered to PMacc C++ from user callables
    position_wf_code: str
    time_wf_code: str
    mask_fourier_code: str

    type_optical_imaging: Literal[True] = True

    @model_validator(mode="after")
    def _check_derived_quantities(self):
        # The band-pass plateau must be non-empty (mirrors the C++ omegaMin < omegaMax requirement).
        if self.d_lambda >= self.central_lambda:
            raise ValueError(
                "d_lambda must be smaller than central_lambda so that the band-pass plateau is non-empty. "
                f"You gave {self.d_lambda=} and {self.central_lambda=}."
            )
        return self


#: The per-simulation (compile-time ``params::``) values, shared by every
#: instance because the plugin's mask functions and constants are compiled in.
COMPILE_TIME_FIELDS = (
    "t_res",
    "x_res",
    "y_res",
    "numerical_aperture",
    "central_lambda",
    "d_lambda",
    "t_wf_buffer",
    "pos_wf_size",
    "numerical_aperture_wf_size",
    "d_lambda_wf",
)

#: The rendered mask-function bodies (compile-time too, see ``COMPILE_TIME_FIELDS``).
MASK_FIELDS = ("position_wf_code", "time_wf_code", "mask_fourier_code")

#: The C++ defaults of ``shadowgraphy.param``, used when no instance is configured.
DEFAULT_COMPILE_TIME_VALUES = {
    "t_res": 2,
    "x_res": 1,
    "y_res": 1,
    "numerical_aperture": 0.23,
    "central_lambda": 750e-9,
    "d_lambda": 20e-9,
    "t_wf_buffer": 32,
    "pos_wf_size": 12,
    "numerical_aperture_wf_size": 0.5,
    "d_lambda_wf": 20e-9,
}


def default_optical_imaging_params() -> dict:
    """The ``shadowgraphy.param`` content for simulations without an imaging diagnostic.

    ``shadowgraphy.param`` is a required part of the PIConGPU input (the plugin
    includes it unconditionally), so the parameters must always be rendered even
    when no ``OpticalImaging`` diagnostic is configured. In that case the plugin
    is inactive anyway (no ``--shadowgraphy.duration`` on the command line), so
    the constant ``1.0`` masks are harmless.
    """
    return {
        **DEFAULT_COMPILE_TIME_VALUES,
        "position_wf_code": "1.0",
        "time_wf_code": "1.0",
        "mask_fourier_code": "1.0",
    }
