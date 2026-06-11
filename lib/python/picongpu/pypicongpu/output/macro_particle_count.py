"""
# SPDX-FileCopyrightText: Masoud Afshari, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal

from pydantic import BaseModel

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species import Species


class MacroParticleCount(BaseModel):
    species: Species
    period: TimeStepSpec
    type_macroparticlecount: Literal[True] = True
