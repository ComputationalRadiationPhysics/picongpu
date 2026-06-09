# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field


class Random(BaseModel):
    type_random: Literal[True] = True
    ppc: int = Field(gt=0)
    """particles per cell (random layout), >0"""
