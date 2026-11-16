"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from . import ionization as ionization
from .collision import Collision as Collision
from .collision import CollisionalPhysicsSetup as CollisionalPhysicsSetup
from .collision import CollisionNumericsConfig as CollisionNumericsConfig
from .collision import ConstLogCollision as ConstLogCollision
from .collision import DynamicLogCollision as DynamicLogCollision
from .ionization.fieldionization import EnergyConservation as EnergyConservation
from .ionization.fieldionization import IonizationCurrent as IonizationCurrent
from .synchrotron import Synchrotron as Synchrotron


# The union of every interaction type that PIConGPU supports. It is used to type
# the ``Simulation.interactions`` field, so that the standard entry point accepts
# both the standard ``picmistandard.PICMI_FieldIonization`` and PIConGPU's own
# interaction classes. Unions are concern-level only and never re-exported at
# ``picmi``.
AnyInteraction = ionization.IonizationModel | Synchrotron | Collision | CollisionalPhysicsSetup

# The same set as concrete classes, for isinstance-based rejection of interaction
# types that PIConGPU does not support.
SUPPORTED_INTERACTION_TYPES = (ionization.IonizationModel, Synchrotron, Collision, CollisionalPhysicsSetup)
