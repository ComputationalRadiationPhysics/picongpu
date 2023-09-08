"""
This file is part of the PIConGPU.
Copyright 2023-2023 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .operation import Operation
from typeguard import typechecked


@typechecked
class DensityOperation(Operation):
    """
    common interface for all operations that create density
      and the not-placed operation
    """
