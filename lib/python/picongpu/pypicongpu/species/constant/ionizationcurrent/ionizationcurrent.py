# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ..constant import Constant


class IonizationCurrent(Constant):
    """base class for all ionization currents models"""

    picongpu_name: str
    """C++ Code type name of ionizer"""
