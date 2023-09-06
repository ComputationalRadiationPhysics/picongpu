"""
This file is part of the PIConGPU.
Copyright 2023 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre
License: GPLv3+
"""

import typeguard

from .plasmaramp import PlasmaRamp


@typeguard.typechecked
class Exponential(PlasmaRamp):
    """exponential plasma ramp, either up or down"""
    def __init__(self,
                 PlasmaLength: float,
                 PlasmaCutoff: float):
        self.PlasmaLength = PlasmaLength
        self.PlasmaCutoff = PlasmaCutoff

    def check(self) -> None:
        if self.PlasmaLength <= 0:
            raise ValueError("PlasmaLength must be >0")
        if self.PlasmaCutoff < 0:
            raise ValueError("PlasmaCutoff must be >=0")

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "PlasmaLength": self.PlasmaLength,
            "PlasmaCutoff": self.PlasmaCutoff
            }
