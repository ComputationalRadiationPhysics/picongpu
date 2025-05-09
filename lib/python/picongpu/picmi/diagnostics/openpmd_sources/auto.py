"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ....pypicongpu.output.openpmd_sources import Auto as PyPIConGPUAuto
import typeguard
import typing


@typeguard.typechecked
class Auto(SourceBase):
    """
    Default data source for openPMD output

    This class provides a convenient way to dump default simulation data (e.g., all
    particle species and fields) using the openPMD standard, with defaults determined
    by the PIC code in particle-in-cell simulations.

    @param filter Name of a filter to select data contributing to the source.
        Default: None (PIC code-dependent).
    """

    # filter = util.build_typesafe_property(typing.Optional[str])

    def __init__(self, filter: typing.Optional[str] = None):
        self.filter = filter
        self.check()

    def check(self) -> None:
        """
        Validate the filter parameter.

        @throw ValueError If the filter is not a string or None.
        """
        if self.filter is not None and not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self.filter)}")

    def get_as_pypicongpu(self) -> PyPIConGPUAuto:
        """
        Convert this Auto source to a PyPIConGPU Auto source.

        @return A PyPIConGPU Auto instance with the same filter.
        """
        self.check()
        return PyPIConGPUAuto(filter=self.filter)
