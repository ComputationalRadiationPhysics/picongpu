"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .... import pypicongpu

import pydantic
import typeguard
import typing


@typeguard.typechecked
class IonizationModel(pydantic.BaseModel):
    """
    common interface for all ionization models

    @note further configurations may be added by implementations
    """

    MODEL_NAME: str
    """ionization model"""

    ion_species: typing.Any
    """PICMI ion species to apply ionization model for"""

    ionization_electron_species: typing.Any
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

    def check(self):
        # import here to avoid circular import that stems from projecting different species types from PIConGPU onto the same `Species` type in PICMI
        from ... import Species

        assert isinstance(self.ion_species, Species), "ion_species must be an instance of the species object"
        assert isinstance(
            self.ionization_electron_species, Species
        ), "ionization_electron_species must be an instance of the species object"

    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        raise NotImplementedError("abstract base class only!")

    def get_as_pypicongpu(self) -> pypicongpu.species.constant.ionizationmodel.IonizationModel:
        raise NotImplementedError("abstract base class only!")
