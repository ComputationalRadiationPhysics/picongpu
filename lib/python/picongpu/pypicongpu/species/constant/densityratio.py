"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pydantic import Field

from picongpu.pypicongpu.species.constant import Constant


class DensityRatio(Constant):
    """
    factor for weighting when using profiles/deriving
    """

    ratio: float = Field(gt=0.0)
    """factor for weighting calculation"""
