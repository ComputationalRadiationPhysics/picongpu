"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import _FieldIonizationModel

from ..... import pypicongpu
from .....pypicongpu.species.constant import ionizationmodel

import enum


class BSIExtension(enum.Enum):
    StarkShift = 0
    EffectiveZ = 1
    # add additional extensions here


class BSI(_FieldIonizationModel):
    """Barrier Suppression Ionization model"""

    MODEL_NAME: str = "BSI"

    BSI_extensions: tuple[BSIExtension, ...]
    """extension to the BSI model; pass ``()`` for the plain model without extensions"""

    def get_as_pypicongpu(self) -> ionizationmodel.IonizationModel:
        self.check()

        if not self.BSI_extensions:
            return ionizationmodel.BSI(
                ionization_current=self._get_ionization_current(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )

        if len(self.BSI_extensions) > 1:
            pypicongpu.util.unsupported("more than one BSI_extension", self.BSI_extensions)

        if self.BSI_extensions[0] is BSIExtension.StarkShift:
            return ionizationmodel.BSIStarkShifted(
                ionization_current=self._get_ionization_current(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )
        if self.BSI_extensions[0] is BSIExtension.EffectiveZ:
            return ionizationmodel.BSIEffectiveZ(
                ionization_current=self._get_ionization_current(),
                ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
            )
        raise ValueError(f"unknown BSI_extension {self.BSI_extensions[0]}.")
