"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .ionizationmodel import IonizationModel

from .... import pypicongpu

import typeguard


@typeguard.typechecked
class GroundStateIonizationModel(IonizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ion_species.register_requirements(self.get_constants())

    def get_constants(self) -> list[pypicongpu.species.constant.Constant]:
        """get all PyPIConGPU constants required by a ground state ionization model in PIConGPU"""
        self.check()

        Z = self.ion_species.picongpu_element.get_atomic_number()
        assert self.ion_species.charge_state <= Z, f"charge_state must be <= atomic number ({Z})"

        element_properties_const = pypicongpu.species.constant.ElementProperties(
            element=self.ion_species.picongpu_element
        )
        return [element_properties_const]
