"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import FieldIonization

from .....pypicongpu.species.constant.ionizationcurrent import None_
from .....pypicongpu.species.constant.ionizationmodel import ADKLinearPolarization, ADKCircularPolarization

from ..... import pypicongpu

import enum


class ADKVariant(enum.Enum):
    LinearPolarization = 0
    CircularPolarization = 1


class ADK(FieldIonization):
    """Barrier Suppression Ioniztion model"""

    MODEL_NAME: str = "ADK"

    ADK_variant: ADKVariant
    """extension to the BSI model"""

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        if self.ADK_variant is ADKVariant.LinearPolarization:
            return ADKLinearPolarization(ionization_current=None_)
        if self.ADK_variant is ADKVariant.CircularPolarization:
            return ADKCircularPolarization(ionization_current=None_)

        # unknown/unsupported ADK variant
        pypicongpu.util.unsupported(f"ADKVariant {self.ADK_variant}")
