"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path

from pydantic import BaseModel

from picongpu.picmi.species import FilteredSpecies, Species

from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec


class ParticleDump(BaseModel):
    species: Species | FilteredSpecies
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    class Config:
        arbitrary_types_allowed = True

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")
