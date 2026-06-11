"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from enum import Enum

from ...pypicongpu.laser import PolarizationType as PyPIConGPUPolarizationType


class PolarizationType(Enum):
    """represents a polarization of a laser"""

    LINEAR = 1
    CIRCULAR = 2

    def get_as_pypicongpu(self):
        if self == PolarizationType.LINEAR:
            return PyPIConGPUPolarizationType.LINEAR
        if self == PolarizationType.CIRCULAR:
            return PyPIConGPUPolarizationType.CIRCULAR
