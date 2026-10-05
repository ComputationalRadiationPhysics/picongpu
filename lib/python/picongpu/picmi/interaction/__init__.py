"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picmistandard import PICMI_Interaction

from . import ionization
from .synchrotron import Synchrotron
from .collision import Collision, CollisionalPhysicsSetup, ConstLogCollision, DynamicLogCollision


# The union of every interaction type that PIConGPU supports. It is used to type
# the ``Simulation.interactions`` field, so that the standard entry point accepts
# both the standard ``picmistandard.PICMI_FieldIonization`` and PIConGPU's own
# interaction classes.
Interaction = ionization.IonizationModel | Synchrotron | Collision | CollisionalPhysicsSetup

# The same set as concrete classes, for isinstance-based rejection of interaction
# types that PIConGPU does not support.
SUPPORTED_INTERACTION_TYPES = (ionization.IonizationModel, Synchrotron, Collision, CollisionalPhysicsSetup)

__all__ = [
    "Interaction",
    "SUPPORTED_INTERACTION_TYPES",
    "PICMI_Interaction",
    "ionization",
    "Synchrotron",
    "Collision",
    "ConstLogCollision",
    "DynamicLogCollision",
    "CollisionalPhysicsSetup",
]
