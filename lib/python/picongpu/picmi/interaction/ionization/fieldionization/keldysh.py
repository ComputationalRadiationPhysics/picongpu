"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import _FieldIonizationModel

from .....pypicongpu.species.constant import ionizationmodel


class Keldysh(_FieldIonizationModel):
    """Keldysh ionization model"""

    MODEL_NAME: str = "Keldysh"

    def get_as_pypicongpu(self) -> ionizationmodel.IonizationModel:
        self.check()

        return ionizationmodel.Keldysh(
            ionization_current=self._get_ionization_current(),
            ionization_electron_species=self.ionization_electron_species.get_as_pypicongpu(),
        )
