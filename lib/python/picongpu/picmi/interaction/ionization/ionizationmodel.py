"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ...species import Species

import pydantic


class IonizationModel(pydantic.BaseModel):
    """
    common interface for all ionization models

    @note further configurations may be added by implementations
    """

    MODEL_NAME: str
    """ionization model"""

    ion_species: Species
    """ion species to apply ionization model for"""

    ionization_electron_species: Species
    """electron species of which to create macro particle upon ionization"""
