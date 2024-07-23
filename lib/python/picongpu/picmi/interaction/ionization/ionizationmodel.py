"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ...species import Species
from .... import pypicongpu

import pydantic


class IonizationModel(pydantic.BaseModel):
    """
    common interface for all ionization models

    @note further configurations may be added by implementations
    """

    MODEL_NAME: str
    """ionization model"""

    ion_species: Species
    """PICMI ion species to apply ionization model for"""

    ionization_electron_species: Species
    """PICMI electron species of which to create macro particle upon ionization"""

    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        raise NotImplementedError("abstract base class only!")

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        raise NotImplementedError("abstract base class only!")
