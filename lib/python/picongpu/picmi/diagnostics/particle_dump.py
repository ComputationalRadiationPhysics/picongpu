# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from picongpu.picmi.diagnostics.backend_config import BackendConfig, OpenPMDConfig
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species


class ParticleDump(BaseModel):
    species: Species | FilteredSpecies
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")
