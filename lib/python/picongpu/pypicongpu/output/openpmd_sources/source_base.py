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
    @property
    @abstractmethod
    def filter(self) -> typing.Optional[str]:
        # Filter name for particle selection, None if no filter
        pass

    @abstractmethod
    def check(self) -> None:
        # Validate data source parameters
        pass

    @abstractmethod
    def _get_serialized(self) -> typing.Dict:
        # Return serialized representation for rendering
        pass
