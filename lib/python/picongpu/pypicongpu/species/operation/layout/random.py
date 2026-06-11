"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal

from pydantic import BaseModel, Field


class Random(BaseModel):
    type_random: Literal[True] = True
    ppc: int = Field(gt=0)
    """particles per cell (random layout), >0"""
