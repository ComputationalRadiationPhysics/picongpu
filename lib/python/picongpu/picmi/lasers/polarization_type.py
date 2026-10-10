"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from enum import Enum

from ...pypicongpu.laser import PolarizationType as PyPIConGPUPolarizationType


class PolarizationType(Enum):
    """represents a polarization of a laser"""

    LINEAR = 1
    CIRCULAR = 2

    def get_as_pypicongpu(self, *args, **kwargs):
        # Extra arguments (e.g. the grid handed through the laser translation)
        # are irrelevant for the polarization and ignored.
        if self == PolarizationType.LINEAR:
            return PyPIConGPUPolarizationType.LINEAR
        if self == PolarizationType.CIRCULAR:
            return PyPIConGPUPolarizationType.CIRCULAR
