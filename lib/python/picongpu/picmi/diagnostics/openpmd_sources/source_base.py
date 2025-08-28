"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from abc import ABCMeta, abstractmethod
import typeguard
import typing


@typeguard.typechecked
class SourceBase(metaclass=ABCMeta):
    """
    Abstract base class for openPMD data sources in particle-in-cell simulations.

    Defines the interface for data sources output via the openPMD standard,
    such as charge density or other derived attributes.

    Subclasses must implement the filter property, check method, and get_as_pypicongpu method.
    """

    @property
    @abstractmethod
    def filter(self) -> typing.Optional[str]:
        """
        Name of a filter to select particles contributing to the data source.

        Returns
        -------
        str or None
            The filter name, or None if no filter is applied.
        """
        pass

    @abstractmethod
    def check(self) -> None:
        valid_filters = ["all", "custom_filter"]
        if self._filter is not None and not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self._filter)}")
        if self._filter is not None and self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    @abstractmethod
    def get_as_pypicongpu(self) -> typing.Any:
        """
        Convert this data source to a PyPIConGPU equivalent.

        Returns
        -------
        Any
            A PyPIConGPU data source object (e.g., pypicongpu.output.openpmd_source.ChargeDensity).
        """
        pass
