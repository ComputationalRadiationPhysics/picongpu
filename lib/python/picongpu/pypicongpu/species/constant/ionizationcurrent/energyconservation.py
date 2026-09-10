"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationcurrent import IonizationCurrent


class EnergyConservation(IonizationCurrent):
    picongpu_name: str = "EnergyConservation"
