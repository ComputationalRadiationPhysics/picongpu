# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species import Species


class MacroParticleCount(BaseModel):
    species: Species
    period: TimeStepSpec
    type_macroparticlecount: Literal[True] = True
