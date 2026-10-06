"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picmistandard import PICMI_Interaction

from ..groundstateionizationmodel import GroundStateIonizationModel
from .ionizationcurrent import IonizationCurrent

from .....pypicongpu.species.constant.ionizationcurrent import IonizationCurrent as PypicongpuIonizationCurrent
from .....pypicongpu.species.constant.ionizationcurrent import None_


class _FieldIonizationModel(GroundStateIonizationModel, PICMI_Interaction):
    """common interface of all concrete field ionization models

    This is an internal base class: users select a concrete model through the
    public :class:`picongpu.picmi.FieldIonization` standard-facing adapter.
    The concrete models (``ADK``, ``BSI``, ``Keldysh``) derive from it and are
    therefore accepted by the standard ``Simulation.interactions`` field just
    like the standard interaction classes.
    """

    ionization_current: IonizationCurrent | None
    """ionization current for energy conservation of field ionization"""

    def _get_ionization_current(self) -> PypicongpuIonizationCurrent:
        """bridge the ionization current to the pypicongpu model

        None maps to the pypicongpu None_ current (the default no-current
        choice). A concrete current is converted via its get_as_pypicongpu
        method and must result in a pypicongpu ionization current model;
        otherwise an error is raised instead of silently dropping the current.
        """
        if self.ionization_current is None:
            return None_()
        current = (
            self.ionization_current.get_as_pypicongpu()
            if hasattr(self.ionization_current, "get_as_pypicongpu")
            else self.ionization_current
        )
        if not isinstance(current, PypicongpuIonizationCurrent):
            raise ValueError(
                f"Unsupported ionization current {self.ionization_current!r}: it does not convert to a pypicongpu "
                "ionization current model, and silently dropping it would change the physics."
            )
        return current
