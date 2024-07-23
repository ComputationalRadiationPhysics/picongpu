"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel

from .... import pypicongpu


class GroundStateIonizationModel(IonizationModel):
    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        """get all PyPIConGPU constants required by a ground state ionization model in PIConGPU"""
        Z = self.ion_species.picongpu_element.get_atomic_number()
        assert self.ion_species.charge_state <= Z, f"charge_state must be <= atomic number ({Z})"

        element_properties_const = pypicongpu.species.constant.ElementProperties()
        element_properties_const.element = self.ion_species.picongpu_element
        return element_properties_const
