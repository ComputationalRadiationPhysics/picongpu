"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import typing

from pydantic import Field

from picongpu.pypicongpu.species.constant import Constant
from picongpu.pypicongpu.species.constant.ionizationcurrent import IonizationCurrent


class IonizationModel(Constant):
    """
    base class for an ground state only ionization models of an ion species

    Owned by exactly one species.

    Identified by its PIConGPU name.

    PIConGPU term: "ionizer"
    """

    ionizer_picongpu_name: str = Field(alias="picongpu_name")
    """C++ Code type name of ionizer"""

    # no typecheck here -- would require circular imports
    ionization_electron_species: typing.Any
    """species to be used as electrons"""

    ionization_current: typing.Optional[IonizationCurrent] = None
    """ionization current implementation to use"""
