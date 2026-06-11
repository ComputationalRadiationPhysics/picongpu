"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .fieldionization import FieldIonization

from .....pypicongpu.species.constant.ionizationcurrent import None_
from .....pypicongpu.species.constant.ionizationmodel import (
    ADKLinearPolarization,
    ADKCircularPolarization,
    IonizationModel,
)


import enum
import typeguard


@typeguard.typechecked
class ADKVariant(enum.Enum):
    LinearPolarization = 0
    CircularPolarization = 1


@typeguard.typechecked
class ADK(FieldIonization):
    """ADK Tunneling Ionization model"""

    MODEL_NAME: str = "ADK"

    ADK_variant: ADKVariant
    """ADK model variant specification"""

    def get_as_pypicongpu(self) -> IonizationModel:
        self.check()

        if self.ADK_variant is ADKVariant.LinearPolarization:
            return ADKLinearPolarization(
                ionization_current=None_(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )
        if self.ADK_variant is ADKVariant.CircularPolarization:
            return ADKCircularPolarization(
                ionization_current=None_(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )

        # unknown ADK variant
        raise ValueError(f"ADKVariant {self.ADK_variant} is not supported.")
