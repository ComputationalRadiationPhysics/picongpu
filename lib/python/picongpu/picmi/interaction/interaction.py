"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ... import pypicongpu

from .ionization.groundstateionizationmodel import GroundStateIonizationModel, IonizationModel

import picmistandard

import typeguard
import pydantic


@typeguard.typechecked
class Interaction(pydantic.BaseModel):
    """
    Common interface of Particle-In-Cell particle interaction extensions

    e.g. collisions, ionization, nuclear reactions

    This interface is only a semantic interface for typing interactions for storage in the simulation object.
    It does not specify interface requirements for sub classes, since they differ too much.
    """

    ground_state_ionization_model_list: list[GroundStateIonizationModel]
    """
    list of all interaction models that change the charge state of ions

    e.g. field ionization, collisional ionization, ...

    """

    # @todo add Collisions as elastic interaction model, Brian Marre, 2024

    @staticmethod
    def update_constant_list(
        existing_list: list[pypicongpu.species.constant.Constant],
        new_list: list[pypicongpu.species.constant.Constant],
    ) -> None:
        """check if dicts may be merged without overwriting previously set values"""

        new_constant_list = []

        for constant_new in new_list:
            exists_already = False
            for constant in existing_list:
                if type(constant) == type(constant_new):
                    # constant_new already exists in existing constants list
                    exists_already = True

                    if constant != constant_new:
                        # same type of constant but conflicting values
                        raise ValueError(f"Constants {constant} and {constant_new} conflict with each other.")

            if not exists_already:
                new_constant_list.append(constant_new)
            # ignore already existing constants

        # update constant_list
        existing_list.extend(new_constant_list)

    def get_interaction_constants(
        self, picmi_species: picmistandard.PICMI_Species
    ) -> tuple[
        list[pypicongpu.species.constant.Constant],
        dict[IonizationModel, pypicongpu.species.constant.ionizationmodel.IonizationModel],
    ]:
        """get list of all constants required by interactions for the given species"""

        has_ionization = False
        constant_list = []
        ionization_model_conversion = {}
        for model in self.ground_state_ionization_model_list:
            if model.ion_species == picmi_species:
                has_ionization = True
                model_constants = model.get_constants()
                Interaction.update_constant_list(constant_list, model_constants)
                ionization_model_conversion[model] = model.get_as_pypicongpu()

        if has_ionization:
            # add GroundStateIonization constant for entire species
            constant_list.append(
                pypicongpu.species.constant.GroundStateIonization(
                    ionization_model_list=ionization_model_conversion.values()
                )
            )

        # add additional interaction sub groups needing constants here
        return constant_list, ionization_model_conversion

    def fill_in_ionization_electron_species(
        self,
        pypicongpu_by_picmi_species: dict[picmistandard.PICMI_Species, pypicongpu.species.Species],
        ionization_model_conversion_by_type_and_species: dict[
            picmistandard.PICMI_Species,
            None | dict[IonizationModel, pypicongpu.species.constant.ionizationmodel.IonizationModel],
        ],
    ) -> None:
        """
        add ionization models to pypicongpu species

        In PICMI ionization is defined as a list ionization models owned by an interaction object which in turn is a
        member of the simulation, with each ionization model storing its PICMI ion and PICMI ionization electron
        species.

        In contrast in PyPIConGPU each ion PyPIConGPU species owns a list of ionization models, each storing its
        PyPIConGPU ionization electron species.

        This creates the problem that upon translation of the PICMI species to an PyPIConGPU species the PyPIConGPU
        ionization electron species might not exist yet.

        Therefore we leave the ionization electron unspecified upon species creation and fill it in from the PICMI
        simulation ionization model list later.

        (And because python uses pointers, this will be applied to the existing species objects passed in
            pypicongpu_by_picmi_species)
        """

        # ground state ionization model
        for species, ionization_model_conversion in ionization_model_conversion_by_type_and_species.items():
            if ionization_model_conversion is not None:
                for picmi_ionization_model, pypicongpu_ionization_model in ionization_model_conversion.items():
                    try:
                        pypicongpu_ionization_electron_species = pypicongpu_by_picmi_species[
                            picmi_ionization_model.ionization_electron_species
                        ]
                    except KeyError:
                        raise ValueError(
                            f"Ionization electron species of {picmi_ionization_model} not known to simulation.\n"
                            + f"Please add species {picmi_ionization_model.ionization_electron_species.name} to"
                            + " the simulation."
                        )
                    pypicongpu_ionization_model.ionization_electron_species = pypicongpu_ionization_electron_species

    def __has_ground_state_ionization(self, species) -> bool:
        """does at least one ground state ionization model list species as ion species?"""

        for ionization_model in self.ground_state_ionization_model_list:
            if species == ionization_model.ion_species:
                return True
        return False

    def has_ionization(self, species) -> bool:
        """does at least one ionization model list species as ion species?"""
        from ..species import Species

        assert isinstance(species, Species)

        # add additional groups of ionization models here
        ionization_configured = self.__has_ground_state_ionization(species)
        return ionization_configured
