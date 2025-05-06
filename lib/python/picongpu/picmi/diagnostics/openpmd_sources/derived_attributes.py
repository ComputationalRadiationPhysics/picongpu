"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ...pypicongpu.output.openpmd_source import DerivedAttributes as PyPIConGPUDerivedAttributes
import typeguard
import typing


@typeguard.typechecked
class DerivedAttributes(SourceBase):
    """
    Aggregated derived attributes data source for openPMD output

    Enables all particle-to-grid derived attributes (e.g., density, charge) for openPMD output
    in particle-in-cell simulations, with defaults determined by the PIC code.

    @param filter Name of a filter to select data. Default: None (PIC code-dependent).
    """

    def __init__(self, filter: typing.Optional[str] = None):
        self.filter = filter
        self.check()

    def check(self) -> None:
        """
        Validate the filter parameter.

        @throw ValueError If filter is not a string or None.
        """
        if self.filter is not None and not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self.filter)}")

    def get_as_pypicongpu(self) -> PyPIConGPUDerivedAttributes:
        """
        Convert to a PyPIConGPU DerivedAttributes source.

        @return A PyPIConGPU DerivedAttributes instance with the same filter.
        """
        self.check()
        return PyPIConGPUDerivedAttributes(filter=self.filter)
