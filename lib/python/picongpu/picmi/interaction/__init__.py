# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from . import ionization
from .synchrotron import Synchrotron
from .collision import Collision, CollisionalPhysicsSetup, ConstLogCollision, DynamicLogCollision

Interaction = ionization.IonizationModel | Synchrotron | Collision | CollisionalPhysicsSetup

__all__ = [
    "Interaction",
    "ionization",
    "Synchrotron",
    "Collision",
    "ConstLogCollision",
    "DynamicLogCollision",
    "CollisionalPhysicsSetup",
]
