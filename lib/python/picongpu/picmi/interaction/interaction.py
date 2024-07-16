"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
from .ionization import IonizationModel


class Interaction(pydantic.BaseModel):
    """
    Common interface of Particle-In-Cell particle interaction extensions

    e.g. collisions, ionization, nuclear reactions

    This interface is only a semantic interface for typing interactions for storage in the simulation object.
    It does not specify interface requirements for sub classes, since they differ too much.
    """

    Ionization: list[IonizationModel]
    """
    list of all interaction models that change the charge state of ions

    e.g. field ionization, collisional ionization, ...

    """

    # @todo add Collisions as elastic interaction model, Brian Marre, 2024
