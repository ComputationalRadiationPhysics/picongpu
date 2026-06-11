"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from functools import lru_cache
from pathlib import Path


@lru_cache
def path():
    return Path(__file__).parent.absolute()
