"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Pawel Ordyna, Masoud Afshari
License: GPLv3+
"""

import typeguard
from typing import Union


from ...pypicongpu.output.auto import Auto as PyPIConGPUAuto
from ..copy_attributes import default_converts_to
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUAuto)
@typeguard.typechecked
class Auto:
    """
    Specifies the parameters for the Auto output.

    Parameters
    ----------
    period: int or TimeStepSpec
        Number of simulation steps between consecutive outputs.
        Unit: steps (simulation time steps).
    """

    def __init__(self, period: Union[int, TimeStepSpec]) -> None:
        if not isinstance(period, (int, TimeStepSpec)):
            raise TypeError("period must be an integer or TimeStepSpec")
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec[::period]("steps") if period > 0 else TimeStepSpec([])("steps")
        else:
            self.period = period

    def check(self, *args, **kwargs):
        """Validate that period is a valid TimeStepSpec."""
        if not isinstance(self.period, TimeStepSpec):
            raise TypeError("period must be a TimeStepSpec")
