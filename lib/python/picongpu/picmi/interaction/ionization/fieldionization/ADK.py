"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .fieldionization import FieldIonization

import enum


class ADKVariant(enum.Enum):
    LinearPolarization = 0
    CircularPolarization = 1


class ADK(FieldIonization):
    """Barrier Suppression Ioniztion model"""

    MODEL_NAME: str = "ADK"

    ADK_variant: ADKVariant
    """extension to the BSI model"""
