"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, Field, ConfigDict

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to

from lasy.laser import Laser as LasyLaser


@default_converts_to(laser.FromLasyLaser)
class FromLasyLaser(BaseModel):
    """PICMI object for FromLasyLaser via FromOpenPMDPulseLaser"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    propagation_direction: list[float]
    polarization_direction: list[float]
    time_offset_si: float
    file_path: str
    lasyLaser: LasyLaser
    iteration: int = 0
    Nt: int | None = None
    Nx: int | None = None
    Ny: int | None = None
    points_between_r: float = 1.0
    forced_dt: float | None = None
    data_step: int = 1
    append: bool = False
    # make sure to always place Huygens-surface inside PML-boundaries,
    # default is valid for standard PMLs
    # @todo create check for insufficient dimension
    # @todo create check in simulation for conflict between PMLs and
    # Huygens-surfaces
    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]]
    )
