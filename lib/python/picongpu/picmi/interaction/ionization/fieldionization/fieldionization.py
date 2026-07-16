"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ..groundstateionizationmodel import GroundStateIonizationModel
from .ionizationcurrent import IonizationCurrent


class FieldIonization(GroundStateIonizationModel):
    """common interface of all field ionization models"""

    ionization_current: IonizationCurrent | None
    """ionization current for energy conservation of field ionization"""
