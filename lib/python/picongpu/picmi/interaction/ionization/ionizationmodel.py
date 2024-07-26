"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ...species import Species
from .... import pypicongpu

import pydantic
import typeguard


@typeguard.typechecked
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

    def __hash__(self):
        """custom hash function for indexing in dicts"""
        hash_value = hash(type(self))

        for value in self.__dict__.values():
            try:
                if value is not None:
                    hash_value += hash(value)
            except TypeError:
                print(self)
                print(type(self))
                raise TypeError
        return hash_value

    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        raise NotImplementedError("abstract base class only!")

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        raise NotImplementedError("abstract base class only!")
