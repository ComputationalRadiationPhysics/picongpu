"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .constant import Constant
from .ionizationmodel import IonizationModel, IonizationModelGroups

import pydantic
import typing


class GroundStateIonization(Constant, pydantic.BaseModel):
    ionization_model_list: list[IonizationModel]
    """list of ground state only ionization models to apply for the species"""

    def get(self):
        return self.ionization_model_list

    def __hash__(self) -> int:
        return_hash_value = hash(type(self))
        for model in self.ionization_model_list:
            return_hash_value += hash(model)
        return return_hash_value

    def check(self) -> None:
        # check that at least one ionization model in list
        if len(self.ionization_model_list) == 0:
            raise ValueError("at least one ionization model must be specified if ground_state_ionization is not none.")

        # call check() all ionization models
        for ionization_model in self.ionization_model_list:
            ionization_model.check()

        # check that no ionization model group is represented more than once
        groups = IonizationModelGroups().get_by_group().keys()

        type_already_present = {}
        for group in groups:
            type_already_present[group] = False

        by_model = IonizationModelGroups().get_by_model()
        for ionization_model in self.ionization_model_list:
            group: str = by_model[type(ionization_model)]
            if type_already_present[group]:
                raise ValueError(f"ionization model group already represented: {group}")
            else:
                type_already_present[group] = True

    def get_species_dependencies(self) -> list[type]:
        """get all species one of the ionization models in ionization_model_list depends on"""

        total_species_dependencies = []
        for ionization_model in self.ionization_model_list:
            species_dependencies = ionization_model.get_species_dependencies()
            for species in species_dependencies:
                if species not in total_species_dependencies:
                    total_species_dependencies.append(species)

        return total_species_dependencies

    def get_attribute_dependencies(self) -> list[type]:
        """get all attributes one of the ionization models in ionization_model_list depends on"""
        total_attribute_dependencies = []
        for ionization_model in self.ionization_model_list:
            attribute_dependencies = ionization_model.get_attribute_dependencies()
            for attribute in attribute_dependencies:
                if attribute not in total_attribute_dependencies:
                    total_attribute_dependencies.append(attribute)

        return total_attribute_dependencies

    def get_constant_dependencies(self) -> list[type]:
        """get all constants one of the ionization models in ionization_model_list depends on"""
        total_constant_dependencies = []
        for ionization_model in self.ionization_model_list:
            constant_dependencies = ionization_model.get_constant_dependencies()
            for constant in constant_dependencies:
                if constant not in total_constant_dependencies:
                    total_constant_dependencies.append(constant)

        return total_constant_dependencies

    def _get_serialized(self) -> dict[str, list[dict[str, typing.Any]]]:
        self.check()

        list_serialized = []
        for ionization_model in self.ionization_model_list:
            list_serialized.append(ionization_model.get_generic_rendering_context())

        return {"ionization_model_list": list_serialized}
