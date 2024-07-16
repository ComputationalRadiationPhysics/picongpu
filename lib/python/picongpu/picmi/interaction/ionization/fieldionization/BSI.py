"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import FieldIonization

import enum


class BSIExtension(enum.Enum):
    StarkShift = 0
    EffectiveZ = 1
    # consider_excitation = 2
    # add additional features here


class BSI(FieldIonization):
    """Barrier Suppression Ioniztion model"""

    MODEL_NAME: str = "BSI"

    BIS_extensions: list[BSIExtension]
    """extension to the BSI model"""
