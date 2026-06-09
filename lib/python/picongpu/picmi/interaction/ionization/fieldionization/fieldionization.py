# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
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
