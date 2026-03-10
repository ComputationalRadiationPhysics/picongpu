"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Julian Lenz
License: GPLv3+
"""

import logging
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PlainSerializer,
    computed_field,
    model_validator,
)


class PolarizationType(Enum):
    """represents a polarization of a laser (for PIConGPU)"""

    LINEAR = "Linear"
    CIRCULAR = "Circular"


def _get_huygens_surface_serialized(huygens_surface_positions) -> dict:
    """Serialize huygens surface positions for all laser types"""
    return {
        "row_x": {
            "negative": huygens_surface_positions[0][0],
            "positive": huygens_surface_positions[0][1],
        },
        "row_y": {
            "negative": huygens_surface_positions[1][0],
            "positive": huygens_surface_positions[1][1],
        },
        "row_z": {
            "negative": huygens_surface_positions[2][0],
            "positive": huygens_surface_positions[2][1],
        },
    }


class _Component(BaseModel):
    component: float

    def __eq__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.component == other
        return super().__eq__(other)


def validate_component_vector(value):
    try:
        return [_Component(component=c) for c in value]
    except Exception:
        return value


class _BaseLaser(BaseModel):
    """Base class for all laser types with common properties and serialization logic"""

    # Common properties for all lasers
    propagation_direction: Annotated[
        tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)
    ]
    """propagation direction (normalized vector)"""
    polarization_direction: Annotated[
        tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)
    ]
    """direction of polarization (normalized vector)"""
    polarization_type: PolarizationType
    """laser polarization"""
    wave_length_si: float = Field(alias="wavelength", gt=0.0)
    """wave length in m"""
    pulse_duration_si: float = Field(alias="duration", gt=0.0)
    """duration in s (1 sigma)"""
    focus_pos_si: Annotated[tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)] = (
        Field(alias="focal_position")
    )
    """focus position vector in m"""
    phase: float = Field(alias="phi0")
    """phi0 in rad, periodic in 2*pi"""
    E0_si: float = Field(alias="E0", gt=0.0)
    """E0 in V/m"""
    pulse_init: float = Field(ge=0.0)
    """laser will be initialized pulse_init times of duration (unitless)"""

    # Huygens surface position (common to all lasers)
    huygens_surface_positions: Annotated[list[list[int]], PlainSerializer(_get_huygens_surface_serialized)]
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""

    def _get_common_serialized_fields(self) -> dict:
        """Get all common serialized fields for lasers"""
        return self.model_dump(mode="json")


def all_ge(values, than_value):
    if any(wrong := [x < than_value for x in values]):
        logging.warning(f"All {values=} should be greater or equal {than_value=}. The following are {wrong=}.")
    return values


def serialise_laguerre(values, suffix):
    return [{f"single_laguerre_{suffix}": x} for x in values]


class GaussianLaser(_BaseLaser):
    """
    PIConGPU Gaussian Laser

    Holds Parameters to specify a gaussian laser
    """

    type_gaussian: Literal[True] = True

    waist_si: float = Field(alias="waist", gt=0.0)
    """beam waist in m"""
    laguerre_modes: Annotated[list[_Component], BeforeValidator(validate_component_vector)] = Field(min_length=1)
    """array containing the magnitudes of radial Laguerre-modes"""
    laguerre_phases: Annotated[list[_Component], BeforeValidator(validate_component_vector)] = Field(min_length=1)
    """array containing the phases of radial Laguerre-modes"""

    @computed_field
    def modenumber(self) -> int:
        return len(self.laguerre_modes) - 1

    @model_validator(mode="after")
    def check(self):
        if len(self.laguerre_phases) != len(self.laguerre_modes):
            raise ValueError("Laguerre modes and Laguerre phases MUST BE arrays of equal length.")
        return self


class PlaneWaveLaser(_BaseLaser):
    """
    PIConGPU Plane Wave Laser

    Holds Parameters to specify a plane wave laser
    """

    type_planewave: Literal[True] = True
    laser_nofocus_constant_si: float
    """constant for plane wave laser without focus (unitless)"""


class DispersivePulseLaser(_BaseLaser):
    """
    PIConGPU Dispersive Pulse Laser

    Holds Parameters to specify a dispersive Gaussian laser pulse with dispersion parameters
    """

    type_dispersive: Literal[True] = True

    waist_si: float = Field(alias="waist")
    """beam waist in m"""
    spectral_support: float
    """width of the spectral support for the discrete Fourier transform [none]"""
    sd_si: float
    """spatial dispersion in focus [m*s]"""
    ad_si: float
    """angular dispersion in focus [rad*s]"""
    gdd_si: float
    """group velocity dispersion in focus [s^2]"""
    tod_si: float
    """third order dispersion in focus [s^3]"""


class FromOpenPMDPulseLaser(BaseModel):
    """
    PIConGPU FromOpenPMDPulseLaser

    Holds Parameters to specify a laser pulse from an OpenPMD file
    """

    type_fromOpenPMDPulse: Literal[True] = True

    propagation_direction: Annotated[
        tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)
    ]
    """propagation direction (normalized vector)"""
    polarization_direction: Annotated[
        tuple[_Component, _Component, _Component], BeforeValidator(validate_component_vector)
    ]
    """direction of polarization (normalized vector)"""
    file_path: str
    """File path to the OpenPMD file containing the pulse data"""
    iteration: int
    """Iteration in the OpenPMD file to use"""
    dataset_name: str
    """Name of the dataset in the OpenPMD file containing the pulse data"""
    datatype: str
    """Data type of the pulse data"""
    time_offset_si: float
    """Time offset in seconds to apply to the pulse data [s]"""
    polarisationAxisOpenPMD: str
    """Polarization axis name in the OpenPMD file"""
    propagationAxisOpenPMD: str
    """Propagation axis name in the OpenPMD file"""
    huygens_surface_positions: Annotated[list[list[int]], PlainSerializer(_get_huygens_surface_serialized)]
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""


class TWTSLaser(_BaseLaser):
    """
    PIConGPU TWTSLaser

    Holds Parameters to specify a TWTS laser pulse
    """

    type_twts: Literal[True] = True

    waist_si: float = Field(alias="waist")
    """beam waist in m"""
    laserIncidenceAngle: float
    """Laser incident angle [rad] denoting the mean laser phase
       propagation direction with respect to the y-axis"""
    laserIncidenceAnglePositive: bool
    """Is the laser incidence angle positive?"""
    polarizationAngle: float
    """Linear laser polarization direction
       parameterized as a rotation angle [rad]
       of the x-direction around the mean
       laser phase propagation direction"""
    beta0: float
    """speed of focal region normalized to the vacuum speed of light [dimensionless]"""
    time_offset_si: float
    """time offset to apply to the pulse [s]"""
    focus_lateral_offset_si: float
    """Offset from the middle of the simulation domain
       to the laser focus in z-direction [m]."""
    windowStart: float
    """First time step number [#] at which the laser starts to be gradually switched on using a Blackman-Nuttall window"""
    windowEnd: float
    """Final time step number [#] after gradually switching off the laser using a Blackman-Nuttall window"""
    windowLength: float
    """Denotes the respective switching duration by half a Blackman-Nuttall window in number of time steps unit [#]"""
    huygens_surface_positions: Annotated[list[list[int]], PlainSerializer(_get_huygens_surface_serialized)]
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""


AnyLaser = DispersivePulseLaser | FromOpenPMDPulseLaser | GaussianLaser | PlaneWaveLaser | TWTSLaser
