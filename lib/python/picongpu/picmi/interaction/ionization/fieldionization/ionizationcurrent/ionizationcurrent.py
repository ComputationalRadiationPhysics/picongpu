"""
# SPDX-FileCopyrightText: Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import pydantic
import typeguard


@typeguard.typechecked
class IonizationCurrent(pydantic.BaseModel):
    """common interface of all ionization current models"""

    MODEL_NAME: str
