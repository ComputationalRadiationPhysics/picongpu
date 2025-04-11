"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import typeguard

from .plasmaramp import PlasmaRamp


@typeguard.typechecked
class None_(PlasmaRamp):
    """no plasma ramp, either up or down"""

    _name = "none"

    def check(self):
        pass

    def _get_serialized(self) -> dict | None:
        return None
