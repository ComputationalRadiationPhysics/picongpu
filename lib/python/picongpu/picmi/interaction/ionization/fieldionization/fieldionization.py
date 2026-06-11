"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from ..groundstateionizationmodel import GroundStateIonizationModel
from .ionizationcurrent import IonizationCurrent

import typing
import typeguard


@typeguard.typechecked
class FieldIonization(GroundStateIonizationModel):
    """common interface of all field ionization models"""

    ionization_current: typing.Optional[IonizationCurrent]
    """ionization current for energy conservation of field ionization"""
