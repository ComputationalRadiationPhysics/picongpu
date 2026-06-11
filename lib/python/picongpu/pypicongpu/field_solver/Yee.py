"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pydantic import BaseModel, computed_field
from ..rendering import RenderedObject


class YeeSolver(RenderedObject, BaseModel):
    """
    Yee solver as defined by PIConGPU

    note: has no parameters
    """

    @computed_field
    def name(self) -> str:
        return "Yee"
