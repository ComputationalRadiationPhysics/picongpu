# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import Constant
from picongpu.pypicongpu.species.util import Element


class ElementProperties(Constant):
    """
    represents constants associated to a chemical element

    Produces PIConGPU atomic number and ionization energies.

    Note: Not necessarily all of the generated properties will be required
    during runtime. However, this is left to the compiler to optimize (which
    is a core concept of PIConGPU).
    """

    element: Element
    """represented chemical element"""
