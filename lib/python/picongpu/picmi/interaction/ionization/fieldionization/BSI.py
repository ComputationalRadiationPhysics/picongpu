"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import FieldIonization

from .....pypicongpu.species.constant.ionizationcurrent import None_
from .....pypicongpu.species.constant import ionizationmodel

from ..... import pypicongpu

import enum


class BSIExtension(enum.Enum):
    StarkShift = 0
    EffectiveZ = 1
    # consider_excitation = 2
    # add additional features here


class BSI(FieldIonization):
    """Barrier Suppression Ioniztion model"""

    MODEL_NAME: str = "BSI"

    BSI_extensions: list[BSIExtension]
    """extension to the BSI model"""

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        if self.BSI_extensions == []:
            return ionizationmodel.BSI(ionization_current=None_)

        if self.BSI_extensions == [BSIExtension.StarkShift]:
            return ionizationmodel.BSIStarkShifted(ionization_current=None_)
        if self.BSI_extensions == [BSIExtension.EffectiveZ]:
            return ionizationmodel.BSIEffectiveZ(ionization_current=None_)

        if len(self.BSI_extensions) > 1:
            pypicongpu.util.unsupported("more than one BSI_extension")
        else:
            pypicongpu.util.unsupported(f"unknown BSI_extension {self.BSI_extensions[0]}")
