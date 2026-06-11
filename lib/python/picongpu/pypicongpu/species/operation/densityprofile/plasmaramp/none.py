"""
# SPDX-FileCopyrightText: Kristin Tippey, Brian Edward Marre, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal
from pydantic import BaseModel


class None_(BaseModel):
    """no plasma ramp, either up or down"""

    type_none: Literal[True] = True
