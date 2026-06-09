# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Data structures to specify and initialize particle species.

Note that the data structure (the classes) here use a different architecure
than both (!) PIConGPU and PICMI.

Please refer to the documentation for a deeper discussion.
"""

from . import operation
from . import attribute
from . import constant

from .species import Species

__all__ = [
    "Species",
    "attribute",
    "constant",
    "operation",
]
