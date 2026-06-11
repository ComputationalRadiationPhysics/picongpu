"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .ionizationcurrent import IonizationCurrent

import typeguard


@typeguard.typechecked
class None_(IonizationCurrent):
    picongpu_name: str = "None"
