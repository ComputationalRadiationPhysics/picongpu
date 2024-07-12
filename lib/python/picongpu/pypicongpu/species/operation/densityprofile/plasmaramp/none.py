"""
This file is part of PIConGPU.
Copyright 2023 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre
License: GPLv3+
"""

import typeguard

from .plasmaramp import PlasmaRamp


@typeguard.typechecked
class None_(PlasmaRamp):
    """no plasma ramp, either up or down"""

    def __init__(self):
        # just overwriting the base class method
        pass

    def check(self) -> None:
        return

    def _get_serialized(self) -> dict | None:
        return None
