"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .... import util
from .source_base import SourceBase
from ...pypicongpu.output.openpmd_source import ChargeDensity as PyPIConGPUChargeDensity
import typeguard
import typing


@typeguard.typechecked
class ChargeDensity(SourceBase):
    """
    Represents the charge density data source for openPMD output in particle-in-cell simulations.

    This class defines the charge density field, derived from particle species at runtime,
    which can be output using the openPMD standard. An optional filter can be applied to select
    which particles contribute to the charge density.

    Parameters
    ----------
    filter: str, optional
        Name of a filter to select particles contributing to the charge density.
        Default: None (all valid particles).
    """

    filter = util.build_typesafe_property(typing.Optional[str])

    def __init__(self, filter: typing.Optional[str] = None):
        self.filter = filter
        self.check()

    def check(self) -> None:
        """
        Validate the filter parameter.

        Raises
        ------
        ValueError
            If the filter is not a string or None.
        """
        if self.filter is not None and not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self.filter)}")

    def get_as_pypicongpu(self) -> PyPIConGPUChargeDensity:
        """
        Convert this ChargeDensity to a PyPIConGPU ChargeDensity object.

        Returns
        -------
        PyPIConGPUChargeDensity
            A PyPIConGPU ChargeDensity instance with the same filter.
        """
        self.check()
        return PyPIConGPUChargeDensity(filter=self.filter)
