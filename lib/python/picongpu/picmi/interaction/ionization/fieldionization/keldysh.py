"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import FieldIonization

from .....pypicongpu.species.constant.ionizationcurrent import None_
from .....pypicongpu.species.constant import ionizationmodel

import typeguard


@typeguard.typechecked
class Keldysh(FieldIonization):
    """Barrier Suppression Ioniztion model"""

    MODEL_NAME: str = "Keldysh"

    def get_as_pypicongpu(self) -> ionizationmodel.IonizationModel:
        self.check()

        return ionizationmodel.Keldysh(ionization_current=None_())
