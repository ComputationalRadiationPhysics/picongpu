"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr

from picongpu.pypicongpu.output.plugin import Plugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species import Species


class MacroParticleCount(Plugin, BaseModel):
    species: Species
    period: TimeStepSpec
    _name: str = PrivateAttr("macroparticlecount")
