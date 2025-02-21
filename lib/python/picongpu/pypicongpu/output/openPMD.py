"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Optional
from .. import util
from ..rendering import RenderedObject

import typeguard


@typeguard.typechecked
class OpenPMD(RenderedObject):
    """
    Class to configure openPMD output.

    This class requires a period (in time steps) and will enable OpenPMD output.
    """

    period = util.build_typesafe_property(int)
    """period to print data at"""

    source = util.build_typesafe_property(str)
    """data sources and filters to dump"""

    range_ = util.build_typesafe_property(str)
    """contiguous range of cells per dimension to dump"""

    file_name = util.build_typesafe_property(str)
    """file name for OpenPMD output"""

    ext = util.build_typesafe_property(str)
    """openPMD filename extension"""

    infix = util.build_typesafe_property(Optional[str])
    """openPMD filename infix"""

    json_ = util.build_typesafe_property(Optional[str])
    """backend-specific parameters for openPMD backends in JSON format"""

    jsonRestart = util.build_typesafe_property(Optional[str])
    """backend-specific parameters for openPMD backends in JSON format for restarting from a checkpoint"""

    dataPreparationStrategy = util.build_typesafe_property(Optional[str])
    """strategy for preparation of particle data"""

    toml = util.build_typesafe_property(Optional[str])
    """configure the openPMD plugin via a TOML file"""

    particleIOChunkSize = util.build_typesafe_property(Optional[int])
    """particle data will be written in chunks of the given size"""

    writeAccess = util.build_typesafe_property(Optional[str])
    """openPMD Access mode for file writing"""

    def check(self) -> None:
        """
        validate attributes

        if ok pass silently, otherwise raises error

        :raises ValueError: period is non-negative integer
        :raises ValueError: file_name is empty string
        :raises ValueError: ext is empty string
        """
        if 1 > self.period:
            raise ValueError("period must be non-negative integer")
        if not self.file_name:
            raise ValueError("file_name cannot be empty")
        if not self.ext:
            raise ValueError("ext cannot be empty")

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "period": self.period,
            "source": self.source,
            "range": self.range_,
            "file_name": self.file_name,
            "ext": self.ext,
            "infix": self.infix,
            "json": self.json_,
            "jsonRestart": self.jsonRestart,
            "dataPreparationStrategy": self.dataPreparationStrategy,
            "toml": self.toml,
            "particleIOChunkSize": self.particleIOChunkSize,
            "writeAccess": self.writeAccess,
        }
