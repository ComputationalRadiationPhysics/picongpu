"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pydantic import model_validator

from .constant import Constant
from .ionizationmodel import IonizationModel, IonizationModelGroups


class GroundStateIonization(Constant):
    ionization_model_list: list[IonizationModel]
    """list of ground state only ionization models to apply for the species"""

    @model_validator(mode="after")
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
        return self
