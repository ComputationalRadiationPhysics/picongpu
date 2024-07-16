"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .BSI import BSI
from .BSIeffectiveZ import BSIEffectiveZ
from .BSIstarkshifted import BSIStarkShifted
from .ADKlinearpolarization import ADKLinearPolarization
from .ADKcircularpolarization import ADKCircularPolarization
from .keldysh import Keldysh
from .thomasfermi import ThomasFermi
from .ionizationmodel import IonizationModel

import copy
import typing
import pydantic


class IonizationModelGroups(pydantic.BaseModel):
    """
    grouping of ionization models into sub groups that may not be used at the same time

    every instance of this class is immutable, all method always return copies of the data contained
    """

    by_group: dict[str, list[typing.Type[IonizationModel]]] = {
        "BSI_like": [BSI, BSIEffectiveZ, BSIStarkShifted],
        "ADK_like": [ADKLinearPolarization, ADKCircularPolarization],
        "Keldysh_like": [Keldysh],
        "electronic_collisional_equilibrium": [ThomasFermi],
    }

    def get_by_group(self) -> dict[str, list[typing.Type[IonizationModel]]]:
        return copy.deepcopy(self.by_group)

    def get_by_model(self) -> dict[typing.Type[IonizationModel], str]:
        return_dict: dict[typing.Type[IonizationModel], str] = {}

        for ionization_model_type, list_ionization_model in self.by_group.items():
            for ionization_model in list_ionization_model:
                return_dict[ionization_model] = copy.deepcopy(ionization_model_type)

        return return_dict
