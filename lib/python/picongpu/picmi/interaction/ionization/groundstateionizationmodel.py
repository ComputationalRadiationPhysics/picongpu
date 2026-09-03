"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel

from .... import pypicongpu


class GroundStateIonizationModel(IonizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ion_species.register_requirements(self.get_constants())

    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        """get all PyPIConGPU constants required by a ground state ionization model in PIConGPU"""
        self.check()

        # the initial charge state is validated against the element by
        # pypicongpu's SetChargeState operation when the species is
        # translated.
        element_properties_const = pypicongpu.species.constant.ElementProperties(
            element=self.ion_species.picongpu_element
        )
        return [element_properties_const]
