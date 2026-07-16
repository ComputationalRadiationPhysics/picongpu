"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from pydantic import BaseModel, model_validator

from picongpu.picmi.species import DependsOn, Species
from picongpu.picmi.species_requirements import (
    GroundStateIonizationConstruction,
    SetChargeStateOperation,
)
from picongpu.pypicongpu.species.attribute.boundelectrons import BoundElectrons
from picongpu.pypicongpu.species.util import Element

from picongpu import pypicongpu


class IonizationModel(BaseModel):
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

    @model_validator(mode="after")
    def check(self):
        if not Element.is_element(self.ion_species.particle_type):
            raise ValueError(f"{self.ion_species=} must be an ion.")
        if self.ion_species.picongpu_fixed_charge:
            raise ValueError(
                f"I'm trying hard to ionize here but {self.ion_species.picongpu_fixed_charge=} is getting in the way."
            )
        if self.ion_species.charge_state is None:
            raise ValueError(
                f"Species {self.ion_species.name} configured with ionization but no initial charge state specified, "
                "must be explicitly specified via charge_state."
            )
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ion_species.register_requirements(
            [
                DependsOn(species=self.ionization_electron_species),
                GroundStateIonizationConstruction(ionization_model=self),
                SetChargeStateOperation(species=self.ion_species),
                BoundElectrons(),
            ]
        )

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
