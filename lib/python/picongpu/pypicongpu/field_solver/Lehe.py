# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, computed_field
from ..rendering import RenderedObject


class LeheSolver(RenderedObject, BaseModel):
    """
    Lehe solver as defined by PIConGPU

    note: has no parameters
    """

    @computed_field
    def name(self) -> str:
        return "Lehe<>"
