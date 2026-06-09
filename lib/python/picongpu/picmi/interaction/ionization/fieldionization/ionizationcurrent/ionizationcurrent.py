"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from pydantic import BaseModel
import typeguard


@typeguard.typechecked
class IonizationCurrent(BaseModel):
    """common interface of all ionization current models"""

    MODEL_NAME: str
