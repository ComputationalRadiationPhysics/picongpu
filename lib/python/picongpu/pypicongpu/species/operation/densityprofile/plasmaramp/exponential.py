# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from typing import Literal
from pydantic import BaseModel, Field


class Exponential(BaseModel):
    """exponential plasma ramp, either up or down"""

    type_exponential: Literal[True] = True
    PlasmaLength: float = Field(gt=0.0)
    PlasmaCutoff: float = Field(ge=0.0)
