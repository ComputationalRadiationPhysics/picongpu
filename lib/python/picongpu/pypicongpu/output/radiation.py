"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from enum import Enum
from operator import attrgetter, itemgetter
from typing import Callable, Literal

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
)
from sympy import Expr, Symbol
from sympy.vector import CoordSys3D, Vector

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.rendering.pmaccprinter import PMAccPrinter
from picongpu.pypicongpu.species import Species


class LinearFrequencies(BaseModel):
    """Linear frequency scale configuration."""

    N_omega: int = Field(2048, description="Number of frequency values in linear scale")
    omega_min: float = Field(0.0, description="Minimum frequency in [1/s]")
    omega_max: float = Field(1.06e16, description="Maximum frequency in [1/s]")
    type_linear_frequencies: Literal[True] = True


class LogFrequencies(BaseModel):
    """Logarithmic frequency scale configuration."""

    N_omega: int = Field(2048, description="Number of frequency values in linear scale")
    omega_min: float = Field(1.0e14, description="Minimum frequency in [1/s]")
    omega_max: float = Field(1.0e17, description="Maximum frequency in [1/s]")
    type_log_frequencies: Literal[True] = True


class FrequenciesFromList(BaseModel):
    """Frequency list configuration."""

    N_omega: int = Field(2048, description="Number of frequency values in linear scale")
    list_location: str = Field(description="Path to text file containing frequencies")
    type_frequencies_from_list: Literal[True] = True


FrequencyConfiguration = LinearFrequencies | LogFrequencies | FrequenciesFromList


class FormFactorConfiguration(Enum):
    """Form factor settings for radiation calculation."""

    CIC_3D = "CIC_3D"
    TSC_3D = "TSC_3D"
    PCS_3D = "PCS_3D"
    CIC_1Dy = "CIC_1Dy"
    Gauss_spherical = "Gauss_spherical"
    Gauss_cell = "Gauss_cell"
    incoherent = "incoherent"
    coherent = "coherent"


class WindowFunctionConfiguration(Enum):
    """Window function settings for radiation."""

    Triangle = "Triangle"
    Hamming = "Hamming"
    Triplett = "Triplett"
    Gauss = "Gauss"
    NONE = "None"


def _make_vector(coefficients, basis_vectors=CoordSys3D("e")):
    # In sympy, vectors are represented as linear combinations of basis vectors.
    # The last argument is important.
    # Otherwise Python tries to start from an integer (scalar) 0 which is not well-defined.
    return sum((coeff * vec for coeff, vec in zip(coefficients, basis_vectors)), Vector.zero)


class RadiationObserverConfiguration(BaseModel):
    """Complete observer configuration."""

    N_observer: int = Field(256, description="Total number of observation directions")
    index_to_direction: Callable[[Symbol], tuple[Expr, Expr, Expr]] = Field(exclude=True)

    @field_validator("index_to_direction", mode="after")
    @classmethod
    def _validate_index_to_direction(cls, value):
        index = Symbol("index")
        vec = _make_vector(value(index))
        if vec.magnitude().equals(1):
            return value
        if vec.magnitude().equals(0):
            raise ValueError(f"The index_to_direction expression must be normalisable. You gave: {vec=} with norm 0.")
        return lambda arg: tuple(
            map(itemgetter(2), sorted(vec.normalize().subs(index, arg).components.items(), key=itemgetter(1)))
        )

    @computed_field
    def component_expressions(self) -> dict[str, str]:
        return {
            key: PMAccPrinter().doprint(value) for key, value in zip("xyz", self.index_to_direction(Symbol("index")))
        }


class RadiationConfiguration(BaseModel):
    """Complete radiation plugin configuration."""

    verbose_level: int = Field(
        3,
        description="Verbose level (0=nothing, 1=physics, 2=sim_state, 4=memory, 8=critical)",
    )

    frequencies: FrequencyConfiguration = Field(
        default_factory=LinearFrequencies,
        description="Frequency scale configuration",
    )

    nyquist_factor: float = Field(0.5, description="Nyquist factor (0 < factor < 1)")

    form_factor: FormFactorConfiguration = Field(
        FormFactorConfiguration.Gauss_spherical,
        description="Form factor type for particle charge distribution",
    )


class RadiationPluginConfig(BaseModel):
    """Top-level radiation plugin configuration.

    Combines radiation settings, observer settings, gamma filtering,
    and window function configuration into a single coherent model.
    """

    radiation: RadiationConfiguration = Field(
        default_factory=RadiationConfiguration,
        description="Core radiation plugin configuration",
    )

    observer: RadiationObserverConfiguration = Field(description="Observer configuration for virtual detectors")

    gamma_filter_threshold: float | None = Field(
        None,
        description="Minimum gamma value for particles to be included in radiation calculation",
    )

    window_function: WindowFunctionConfiguration = Field(
        WindowFunctionConfiguration.NONE,
        description="Window function to reduce ringing effects",
    )

    num_accumulation_steps: int = Field(
        0,
        description="Period, after which the calculated radiation data should be dumped to the file system. Default is 0, therefore never. In order to store the radiation data, a value >=1 should be used.",
    )

    last_radiation: bool = Field(
        False,
        description="If set, the radiation spectra summed between the last and the current dump-time-step are stored. Used for a better evaluation of the temporal evolution of the emitted radiation.",
    )

    folder_last_rad: str = Field(
        "lastRad",
        description="Name of the folder, in which the summed spectra for the simulation time between the last dump and the current dump are stored. Default is 'lastRad'.",
    )

    total_radiation: bool = Field(
        False,
        description="If set the spectra summed from simulation start till current time step are stored.",
    )

    folder_total_rad: str = Field(
        "totalRad",
        description="Folder name in which the total radiation spectra, integrated from the beginning of the simulation, are stored. Default 'totalRad'.",
    )

    start: int = Field(
        2,
        description="Time step, at which PIConGPU starts calculating the radiation. Default is 2 in order to get enough history of the particles.",
    )

    end: int = Field(
        0,
        description="Time step, at which the radiation calculation should end. Default: 0 (stops at end of simulation).",
    )

    rad_per_gpu: bool = Field(
        False,
        description="If set, each GPU additionally stores its own spectra without summing over the entire simulation area. This allows for a localization of specific spectral features.",
    )

    folder_rad_per_gpu: str = Field(
        "radPerGPU",
        description="Name of the folder, where the GPU specific spectra are stored. Default: 'radPerGPU'",
    )

    num_jobs: int = Field(
        2,
        description="Number of independent jobs used for the radiation calculation. This option is used to increase the utilization of the device by producing more independent work. This option enables accumulation of data in parallel into multiple temporary arrays, thereby increasing the utilization of the device by increasing the memory footprint. Default: 2",
    )

    open_pmd_suffix: str = Field(
        "_%T_0_0_0.h5",
        description="This sets the suffix for openPMD filename extension and iteration expansion pattern. Default: '_%T_0_0_0.h5'",
    )

    open_pmd_checkpoint_extension: str = Field(
        "h5",
        description="Set filename extension for openPMD checkpoints. Default: 'h5'",
    )

    open_pmd_config: str = Field(
        "{}",
        description="Give JSON/TOML configuration for initializing openPMD. Default: '{}' (no JSON/TOML configuration used)",
    )

    open_pmd_checkpoint_config: str = Field(
        "{}",
        description="Give JSON/TOML configuration for initializing openPMD checkpointing. Default: '{}' (no JSON/TOML configuration used)",
    )

    distributed_amplitude: bool = Field(
        False,
        description="Activate the optional output of distributed amplitudes per MPI rank. in the openPMD output. Default: 0 (deactivated/no additional output)",
    )


class RadiationPlugin(BaseModel):
    type_radiation: Literal[True] = True
    config: RadiationPluginConfig
    species: list[Species]
    period: TimeStepSpec

    @field_validator("period", mode="after")
    @classmethod
    def _validate_period(cls, period):
        if 0 in map(attrgetter("start"), period.specs):
            raise ValueError(f"The radiation plugin cannot produce output at time step 0. You gave {period=}.")
        return period
