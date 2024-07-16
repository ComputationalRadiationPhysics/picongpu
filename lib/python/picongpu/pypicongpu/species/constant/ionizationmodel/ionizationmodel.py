"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ..constant import Constant
from ...attribute import BoundElectrons
from ..ionizationcurrent import IonizationCurrent
from ..elementproperties import ElementProperties

import pydantic
import typing


class IonizationModel(pydantic.BaseModel, Constant):
    """
    base class for an ground state only ionization models of an ion species

    Owned by exactly one species.

    Identified by its PIConGPU name.

    PIConGPU term: "ionizer"
    """

    PICONGPU_NAME: str
    """C++ Code type name of ionizer"""

    # no typecheck here -- would require circular imports
    ionization_electron_species: typing.Any = None
    """species to be used as electrons"""

    ionization_current: typing.Optional[IonizationCurrent] = None
    """ionization current implementation to use"""

    def check(self) -> None:
        """check internal consistency"""

        # import here to avoid circular import
        from ...species import Species
        from ..groundstateionization import GroundStateIonization

        # check ionization electron species is actually pypicongpu species instance
        if not isinstance(self.ionization_electron_species, Species):
            raise TypeError("ionization_electron_species must be of type pypicongpu Species")

        # electron species must not be an ionizable
        if self.ionization_electron_species.has_constant_of_type(GroundStateIonization):
            raise ValueError(
                "used electron species {} must not be ionizable itself".format(self.ionization_electron_species.name)
            )

        # test ionization current set if required
        test = self.ionization_current  # noqa

        # note: do **NOT** check() electron species here
        # -> it is not fully initialized at this point in the initialization
        # (check requires attributes which are added last,
        # but constants are added first)

    def _get_serialized(self) -> dict[str, typing.Any]:
        # do not remove!, always do a check call
        self.check()

        if self.ionization_current is None:
            # case no ionization_current configurable
            return {
                "ionizer_picongpu_name": self.PICONGPU_NAME,
                "ionization_electron_species": self.ionization_electron_species.get_rendering_context(),
                "ionization_current": None,
            }

        # default case
        return {
            "ionizer_picongpu_name": self.PICONGPU_NAME,
            "ionization_electron_species": self.ionization_electron_species.get_rendering_context(),
            "ionization_current": self.ionization_current.get_generic_rendering_context(),
        }

    def get_generic_rendering_context(self) -> dict[str, typing.Any]:
        return IonizationModel(
            PICONGPU_NAME=self.PICONGPU_NAME,
            ionization_electron_species=self.ionization_electron_species,
            ionization_current=self.ionization_current,
        ).get_rendering_context()

    def get_species_dependencies(self) -> list[type]:
        self.check()
        return [self.ionization_electron_species]

    def get_attribute_dependencies(self) -> list[type]:
        return [BoundElectrons]

    def get_constant_dependencies(self) -> list[type]:
        return [ElementProperties]
