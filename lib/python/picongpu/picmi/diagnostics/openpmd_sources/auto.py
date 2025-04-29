"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .... import util
from .source_base import SourceBase
from ...pypicongpu.output.openpmd_source import Auto as PyPIConGPUAuto
import typeguard
import typing


@typeguard.typechecked
class Auto(SourceBase):
    """
    Represents a default data source for openPMD output in particle-in-cell simulations.

    This class specifies a backend-specific default source for dumping simulation data
    using the openPMD standard. For example, in some backends, it may dump all particle
    species and fields.

    Parameters
    ----------
    filter: str, optional
        Name of a filter to select data contributing to the source.
        Default: None (backend-dependent).
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

    def get_as_pypicongpu(self) -> PyPIConGPUAuto:
        """
        Convert this Auto source to a PyPIConGPU Auto source.

        Returns
        -------
        PyPIConGPUAuto
            A PyPIConGPU Auto instance with the same filter.
        """
        self.check()
        return PyPIConGPUAuto(filter=self.filter)
