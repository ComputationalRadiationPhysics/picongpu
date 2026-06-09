# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from typing import Dict, Optional

import typeguard

from picongpu.picmi.copy_attributes import default_converts_to

from ...pypicongpu.output.checkpoint import Checkpoint as PyPIConGPUCheckpoint
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUCheckpoint)
@typeguard.typechecked
class Checkpoint:
    """
    Specifies the parameters for creating checkpoints in PIConGPU simulations.

    This plugin saves simulation state snapshots at specified intervals,
    allowing for simulation restarts or analysis.

    Attention: ** At least one of period or timePeriod must be provided.**

    Parameters
    ----------
    period: TimeStepSpec, optional
        Specify on which time steps to create checkpoints.
        Unit: steps (simulation time steps). Required if timePeriod is not provided.

    timePeriod: int, optional
        Specify the interval in minutes for creating checkpoints.
        Unit: minutes (must be a non-negative integer). Required if period is not provided.

    directory: str, optional
        Directory inside simOutput for writing checkpoints (default: "checkpoints").

    file: str, optional
        Relative or absolute fileset prefix for checkpoint files.

    restart: bool, optional
        If True, restart simulation from the latest checkpoint.

    tryRestart: bool, optional
        If True, restart from the latest checkpoint if available, else start from scratch.

    restartStep: int, optional
        Specific checkpoint step to restart from.

    restartDirectory: str, optional
        Directory inside simOutput containing checkpoints for restart (default: "checkpoints").

    restartFile: str, optional
        Relative or absolute fileset prefix for reading checkpoints.

    restartChunkSize: int, optional
        Number of particles processed in one kernel call during restart.

    restartLoop: int, optional
        Number of times to restart the simulation after it finishes.

    openPMD: Dict, optional
        Dictionary of openPMD-specific settings (e.g., ext, backendConfig, infix).
    """

    def check(self, *args, **kwargs):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        if self.timePeriod is not None and self.timePeriod < 0:
            raise ValueError("timePeriod must be a non-negative integer")
        if self.restartStep is not None and self.restartStep < 0:
            raise ValueError("restartStep must be non-negative")
        if self.restartChunkSize is not None and self.restartChunkSize < 1:
            raise ValueError("restartChunkSize must be positive")
        if self.restartLoop is not None and self.restartLoop < 0:
            raise ValueError("restartLoop must be non-negative")

    def __init__(
        self,
        period: Optional[TimeStepSpec] = None,
        timePeriod: Optional[int] = None,
        directory: Optional[str] = None,
        file: Optional[str] = None,
        restart: Optional[bool] = None,
        tryRestart: Optional[bool] = None,
        restartStep: Optional[int] = None,
        restartDirectory: Optional[str] = None,
        restartFile: Optional[str] = None,
        restartChunkSize: Optional[int] = None,
        restartLoop: Optional[int] = None,
        openPMD: Optional[Dict] = None,
    ):
        self.period = period
        self.timePeriod = timePeriod
        self.directory = directory
        self.file = file
        self.restart = restart
        self.tryRestart = tryRestart
        self.restartStep = restartStep
        self.restartDirectory = restartDirectory
        self.restartFile = restartFile
        self.restartChunkSize = restartChunkSize
        self.restartLoop = restartLoop
        self.openPMD = openPMD
