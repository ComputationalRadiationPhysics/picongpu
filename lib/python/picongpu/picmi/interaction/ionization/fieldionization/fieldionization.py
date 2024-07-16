"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel
from .ionizationcurrent import IonizationCurrent

import typing


class FieldIonization(IonizationModel):
    """common interface of all field ionization models"""

    ionization_current: typing.Optional[IonizationCurrent]
    """ionization current for energy conservation of field ionization"""

    # + all IonizationModel interface requirements
