# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

    def get_as_pypicongpu(self):
        if self == PolarizationType.LINEAR:
            return PyPIConGPUPolarizationType.LINEAR
        if self == PolarizationType.CIRCULAR:
            return PyPIConGPUPolarizationType.CIRCULAR
