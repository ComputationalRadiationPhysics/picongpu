"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import FieldIonization


class Keldysh(FieldIonization):
    """Barrier Suppression Ioniztion model"""

    MODEL_NAME: str = "Keldysh"
