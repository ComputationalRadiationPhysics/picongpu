# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typeguard


@typeguard.typechecked
class IonizationCurrent(pydantic.BaseModel):
    """common interface of all ionization current models"""

    MODEL_NAME: str
