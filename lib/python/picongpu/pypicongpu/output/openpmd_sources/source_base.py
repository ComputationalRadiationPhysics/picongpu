"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...rendering import SelfRegisteringRenderedObject
from abc import ABCMeta, abstractmethod
import typeguard


@typeguard.typechecked
class SourceBase(SelfRegisteringRenderedObject, metaclass=ABCMeta):
    """
    Abstract base class for OpenPMD sources in PIConGPU.
    """

    @property
    @abstractmethod
    def filter(self) -> str:
        """
        Filter name for particle selection.

        Returns
        -------
        str
            Filter name.
        """
        pass

    @abstractmethod
    def check(self) -> None:
        """
        Validate data source parameters.
        """
        pass
