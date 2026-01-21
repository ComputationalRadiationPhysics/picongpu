"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from pydantic import Field

from picongpu.pypicongpu.species.constant import Constant


class DensityRatio(Constant):
    """
    factor for weighting when using profiles/deriving
    """

    ratio: float = Field(gt=0.0)
    """factor for weighting calculation"""
