"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ... import pypicongpu

import picmistandard
import pydantic


class InteractionInterface(pydantic.BaseModel):
    """
    interface for forward declaration
    """

    def get_interaction_constants(
        self, species: picmistandard.PICMI_Species
    ) -> list[pypicongpu.species.constant.Constant]:
        """get list of all constants required by interactions for the given species"""
        raise NotImplementedError("abstract interface for forward declaration only!")

    def fill_in_ionization_electron_species(
        self, pypicongpu_by_picmi_species: dict[picmistandard.PICMI_Species, pypicongpu.species.Species]
    ):
        """add ionization electron species to pypicongpu species' ionization model"""
        raise NotImplementedError("abstract interface for forward declaration only!")
