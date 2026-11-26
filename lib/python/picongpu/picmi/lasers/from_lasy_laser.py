"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Author: Edgar Marquardt
License: GPLv3+
"""

from pydantic import BaseModel, Field, ConfigDict

from ...pypicongpu import laser
from .._copy_attributes import default_converts_to

from lasy.laser import Laser as LasyLaser


@default_converts_to(laser.FromLasyLaser)
class FromLasyLaser(BaseModel):
    """PICMI object for FromLasyLaser via FromOpenPMDPulseLaser"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    propagation_direction: list[float] = Field(..., description="direction of propagation (normalized vector)")
    polarization_direction: list[float] = Field(..., description="direction of polarization (normalized vector)")
    time_offset_si: float = Field(..., description="time offset in seconds")
    file_path: str = Field(
        ...,
        description="File path to the OpenPMD file meant to contain the pulse data. If the file already exists, it will not be overwritten! Use the append option to add to an existing file or remove it yourself.",
    )
    lasyLaser: LasyLaser = Field(
        ..., description="Lasy laser object to be converted to OpenPMD format and used in PIConGPU"
    )
    iteration: int = Field(default=0, description="Iteration number for the laser pulse")
    Nt: int | None = Field(
        default=None,
        description="""Number of time points on which field should be sampled. If it is None,
            the orignal time grid is used, otherwise the field is interpolated on a new grid.""",
    )
    Nx: int | None = Field(
        default=None,
        description="Number of x-points the field should be cut down to. Does not interpolate. Always in the middle.",
    )
    Ny: int | None = Field(
        default=None,
        description="Number of y-points the field should be cut down to. Does not interpolate. Always in the middle.",
    )
    points_between_r: float = Field(
        default=1.0,
        description="""If laser.dim=="rt" the field is converted to xyt to write into the file.
            This argument describes, how many points in x and y directions should be placed
            (interpolated) between two given values in the r direction.""",
    )
    forced_dt: float | None = Field(
        default=None,
        description="""Forces dt to be this value, if possible.
            If forced_dt and Nt is given the resulting section is taken from the end of the field.""",
    )
    data_step: int = Field(
        default=1,
        description="Only saves every (data_step)th data point to the file transversally. The numbers Nx and Ny are still reached.",
    )
    append: bool = Field(default=False, description="Whether to append to an existing file")
    # make sure to always place Huygens-surface inside PML-boundaries,
    # default is valid for standard PMLs
    # Huygens-surfaces
    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]],
        description="Positions of the Huygens surfaces in the simulation box. The default values are valid for standard PMLs.",
    )
