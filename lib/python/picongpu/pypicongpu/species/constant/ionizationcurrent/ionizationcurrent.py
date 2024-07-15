"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ..constant import Constant

import pydantic
import typing


class IonizationCurrent(Constant, pydantic.BaseModel):
    """base class for all ionization currents models"""

    PICONGPU_NAME: str
    """C++ Code type name of ionizer"""

    def check(self) -> None:
        # nothing to check here
        pass

    def _get_serialized(self) -> dict:
        # do not remove!, always check
        self.check()
        return {"picongpu_name": self.PICONGPU_NAME}

    def get_species_dependencies(self):
        return []

    def get_attribute_dependencies(self) -> typing.List[type]:
        return []

    def get_constant_dependencies(self) -> typing.List[type]:
        return []
