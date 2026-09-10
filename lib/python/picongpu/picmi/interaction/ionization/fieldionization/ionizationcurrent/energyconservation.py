"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationcurrent import IonizationCurrent

from ......pypicongpu.species.constant.ionizationcurrent import EnergyConservation as PypicongpuEnergyConservation


class EnergyConservation(IonizationCurrent):
    """energy-conserving ionization current model for field ionization"""

    MODEL_NAME: str = "EnergyConservation"

    def get_as_pypicongpu(self) -> PypicongpuEnergyConservation:
        return PypicongpuEnergyConservation()
