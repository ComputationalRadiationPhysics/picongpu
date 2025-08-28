"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ....pypicongpu.output.openpmd_sources import Auto as PyPIConGPUAuto
import typeguard


@typeguard.typechecked
class Auto(SourceBase):
    """
    Default data source for openPMD output

    This class provides a convenient way to dump default simulation data (e.g., all
    particle species and fields) using the openPMD standard, with defaults determined
    by the PIC code in particle-in-cell simulations.
    """

    def __init__(self, filter: str = "species_all"):
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        """
        Validate the filter parameter.
        """
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    def get_as_pypicongpu(self) -> PyPIConGPUAuto:
        """
        Convert this Auto source to a PyPIConGPU Auto source.
        """
        self.check()
        return PyPIConGPUAuto(filter=self._filter)
