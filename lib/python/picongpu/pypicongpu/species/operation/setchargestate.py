"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..species import Species


class SetChargeState(BaseModel):
    """
    assigns boundElectrons attribute and sets it to the initial charge state

    used for ionization of ions
    """

    species: Species
    """species which will have boundElectrons set"""

    charge_state: int = Field(ge=0)
    """initial ion charge state"""

    type_setchargestate: Literal[True] = True

    @model_validator(mode="after")
    def check(self) -> "SetChargeState":
        element_properties = self.species.constants.element_properties
        if element_properties is not None:
            atomic_number = element_properties.element.get_atomic_number()
            if self.charge_state > atomic_number:
                raise ValueError(
                    f"initial charge state ({self.charge_state}) exceeds the atomic number "
                    f"({atomic_number}) of element {element_properties.element.symbol}"
                )
        return self
