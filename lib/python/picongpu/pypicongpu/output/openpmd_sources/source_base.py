"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...rendering import RenderedObject
from abc import ABCMeta, abstractmethod
import typeguard
import typing


@typeguard.typechecked
class SourceBase(RenderedObject, metaclass=ABCMeta):
    """
    Abstract base class for OpenPMD sources in PIConGPU.
    """

    @property
    @abstractmethod
    def filter(self) -> typing.Optional[str]:
        """
        Filter name for particle selection.

        Returns
        -------
        str or None
            Filter name, or None if no filter is applied.
        """
        pass

    @abstractmethod
    def check(self) -> None:
        """
        Validate data source parameters.
        """
        pass

    def _get_serialized(self) -> typing.Dict:
        """
        Return serialized representation for rendering.

        Returns
        -------
        dict
            Serialized representation including at least 'name'.
        """
        return {"name": self.__class__.__name__.lower()}
