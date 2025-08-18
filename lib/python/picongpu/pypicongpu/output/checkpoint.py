"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .. import util
from .plugin import Plugin
from .timestepspec import TimeStepSpec

import typeguard
import typing
from typing import Optional, Dict


@typeguard.typechecked
class Checkpoint(Plugin):
    period = util.build_typesafe_property(Optional[TimeStepSpec])
    timePeriod = util.build_typesafe_property(Optional[int])
    directory = util.build_typesafe_property(Optional[str])
    file = util.build_typesafe_property(Optional[str])
    restart = util.build_typesafe_property(Optional[bool])
    tryRestart = util.build_typesafe_property(Optional[bool])
    restartStep = util.build_typesafe_property(Optional[int])
    restartDirectory = util.build_typesafe_property(Optional[str])
    restartFile = util.build_typesafe_property(Optional[str])
    restartChunkSize = util.build_typesafe_property(Optional[int])
    restartLoop = util.build_typesafe_property(Optional[int])
    openPMD = util.build_typesafe_property(Optional[Dict])

    _name = "checkpoint"

    def __init__(self):
        self.period = None
        self.timePeriod = None
        self.directory = None
        self.file = None
        self.restart = None
        self.tryRestart = None
        self.restartStep = None
        self.restartDirectory = None
        self.restartFile = None
        self.restartChunkSize = None
        self.restartLoop = None
        self.openPMD = None

    def check(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        if self.period is not None:
            self.period.check()
        if self.timePeriod is not None and self.timePeriod < 0:
            raise ValueError("timePeriod must be non-negative")
        if self.restartStep is not None and self.restartStep < 0:
            raise ValueError("restartStep must be non-negative")
        if self.restartChunkSize is not None and self.restartChunkSize < 1:
            raise ValueError("restartChunkSize must be positive")
        if self.restartLoop is not None and self.restartLoop < 0:
            raise ValueError("restartLoop must be non-negative")

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        self.check()
        serialized = {
            "period": self.period.get_rendering_context() if self.period is not None else None,
            "timePeriod": self.timePeriod,
            "directory": self.directory,
            "file": self.file,
            "restart": self.restart,
            "tryRestart": self.tryRestart,
            "restartStep": self.restartStep,
            "restartDirectory": self.restartDirectory,
            "restartFile": self.restartFile,
            "restartChunkSize": self.restartChunkSize,
            "restartLoop": self.restartLoop,
            "openPMD": self.openPMD,
        }
        return serialized
