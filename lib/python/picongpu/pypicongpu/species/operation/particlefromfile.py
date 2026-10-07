"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..species import Species


class ParticleFromFile(BaseModel):
    """Load the particles of a species from an external openPMD file.

    Unlike :class:`~picongpu.pypicongpu.species.operation.simpledensity.SimpleDensity`,
    this does not create particles from a density profile; it reads the
    positions, momenta and weightings of already-existing macroparticles from an
    openPMD particle file and places them into the domain at ``t=0``.
    """

    species: Species
    """species to be filled from the file"""

    file_path: str
    """path to the openPMD particle file"""

    iteration: int = Field(default=0, ge=0)
    """openPMD iteration to read; defaults to 0"""

    type_particlefromfile: Literal[True] = True
