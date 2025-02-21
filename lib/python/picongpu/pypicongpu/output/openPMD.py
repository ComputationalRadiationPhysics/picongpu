"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .. import util
from ..rendering import RenderedObject

import typeguard


@typeguard.typechecked
class OpenPMD(RenderedObject):
    """
    Class to configure openPMD output.

    This class requires a period (in time steps) and file name.
    """

    period = util.build_typesafe_property(int)
    """period to print data at"""

    file_name = util.build_typesafe_property(str)
    """file name for openPMD output"""

    file_extension = util.build_typesafe_property(str)
    """file extension for openPMD output (e.g., 'bp', 'h5')"""

    def check(self) -> None:
        """
        validate attributes

        if ok pass silently, otherwise raises error

        :raises ValueError: period is non-negative integer
        :raises ValueError: file_name is empty string
        """
        if 1 > self.period:
            raise ValueError("period must be non-negative integer")
        if not self.file_name:
            raise ValueError("file_name cannot be empty")

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "period": self.period,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
        }
