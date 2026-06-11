"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .one_position import OnePosition
from .quiet import Quiet
from .random import Random

AnyLayout = Random | Quiet | OnePosition
__all__ = ["Random", "Quiet", "OnePosition"]
