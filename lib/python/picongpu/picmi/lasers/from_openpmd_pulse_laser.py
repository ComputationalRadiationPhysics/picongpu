"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, Field

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to


@default_converts_to(laser.FromOpenPMDPulseLaser)
class FromOpenPMDPulseLaser(BaseModel):
    """PICMI object for FromOpenPMDPulseLaser"""

    propagation_direction: list[float]
    polarization_direction: list[float]
    time_offset_si: float
    file_path: str
    iteration: int
    dataset_name: str
    datatype: str
    polarisationAxisOpenPMD: str
    propagationAxisOpenPMD: str
    picongpu_huygens_surface_positions: list[list[int]] = Field(default_factory=lambda: [[16, -16], [16, -16], [16, -16]])