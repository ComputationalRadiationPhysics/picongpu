"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic


class IonizationCurrent(pydantic.BaseModel):
    """common interface of all ionization current models"""

    MODEL_NAME: str
