"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .fieldionization import FieldIonization

from ..... import pypicongpu
from .....pypicongpu.species.constant.ionizationcurrent import None_
from .....pypicongpu.species.constant import ionizationmodel

import enum
import typeguard


@typeguard.typechecked
class BSIExtension(enum.Enum):
    StarkShift = 0
    EffectiveZ = 1
    # add additional extensions here


@typeguard.typechecked
class BSI(FieldIonization):
    """Barrier Suppression Ionization model"""

    MODEL_NAME: str = "BSI"

    BSI_extensions: tuple[BSIExtension]
    """extension to the BSI model"""

    def get_as_pypicongpu(self) -> ionizationmodel.IonizationModel:
        self.check()

        if self.BSI_extensions == []:
            return ionizationmodel.BSI(
                ionization_current=None_(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )

        if len(self.BSI_extensions) > 1:
            pypicongpu.util.unsupported("more than one BSI_extension, will use first entry only")

        if self.BSI_extensions[0] is BSIExtension.StarkShift:
            return ionizationmodel.BSIStarkShifted(
                ionization_current=None_(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )
        if self.BSI_extensions[0] is BSIExtension.EffectiveZ:
            return ionizationmodel.BSIEffectiveZ(
                ionization_current=None_(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )
        raise ValueError(f"unknown BSI_extension {self.BSI_extensions[0]}.")
