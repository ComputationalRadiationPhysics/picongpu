"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from ..constant import Constant


class IonizationCurrent(Constant):
    """base class for all ionization currents models"""

    picongpu_name: str
    """C++ Code type name of ionizer"""
