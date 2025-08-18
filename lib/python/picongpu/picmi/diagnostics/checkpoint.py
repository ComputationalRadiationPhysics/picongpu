"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.checkpoint import Checkpoint as PyPIConGPUCheckpoint
from .timestepspec import TimeStepSpec

import typeguard
from typing import Optional, Dict
import warnings


@typeguard.typechecked
class Checkpoint:
    """
    Specifies the parameters for creating checkpoints in PIConGPU simulations.

    This plugin saves simulation state snapshots at specified intervals,
    allowing for simulation restarts or analysis.

    Attention: ** At least one of period or timePeriod must be provided to enable checkpointing.**

    Parameters
    ----------
    period: int or TimeStepSpec, optional
        Number of simulation steps between consecutive checkpoints (e.g., 10 for every 10 steps).
        Use 0 to disable checkpointing.
        Alternatively, a TimeStepSpec can be provided for PIConGPU-specific step selection
        (e.g., TimeStepSpec[5, 10], TimeStepSpec[-10:]).
        Unit: steps (simulation time steps).

    timePeriod: int, optional
        Interval in minutes for creating checkpoints.
        Unit: minutes (must be a non-negative integer).

    directory: str, optional
        Directory inside simOutput for writing checkpoints (default: "checkpoints").

    file: str, optional
        Relative or absolute fileset prefix for checkpoint files.

    restart: bool, optional
        If True, restart simulation from the latest checkpoint.

    tryRestart: bool, optional
        If True, restart from the latest checkpoint if available, else start from scratch.

    restartStep: int, optional
        Specific checkpoint step to restart from (must be non-negative).

    restartDirectory: str, optional
        Directory inside simOutput containing checkpoints for restart (default: "checkpoints").

    restartFile: str, optional
        Relative or absolute fileset prefix for reading checkpoints.

    restartChunkSize: int, optional
        Number of particles processed in one kernel call during restart (must be positive).

    restartLoop: int, optional
        Number of times to restart the simulation after it finishes (must be non-negative).

    openPMD: Dict, optional
        Dictionary of openPMD-specific settings (e.g., ext, json, infix).
    """

    def check(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided to enable checkpointing")
        if self.timePeriod is not None and self.timePeriod < 0:
            raise ValueError("timePeriod must be a non-negative integer")
        if self.restartStep is not None and self.restartStep < 0:
            raise ValueError("restartStep must be non-negative")
        if self.restartChunkSize is not None and self.restartChunkSize < 1:
            raise ValueError("restartChunkSize must be positive")
        if self.restartLoop is not None and self.restartLoop < 0:
            raise ValueError("restartLoop must be non-negative")
        if (
            self.period is not None
            and not self.period.get_as_pypicongpu(1.0, 100).get_rendering_context().get("specs", [])
            and (self.timePeriod is None or self.timePeriod == 0)
        ):
            warnings.warn(
                "Checkpoint is disabled because period is set to 0 or an empty TimeStepSpec and timePeriod is None or 0"
            )

    def __init__(
        self,
        period: Optional[int | TimeStepSpec] = None,
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
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec[::period] if period > 0 else TimeStepSpec()
        else:
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

    def get_as_pypicongpu(
        self,
        pypicongpu_by_picmi_species: Dict,
        time_step_size: float,
        num_steps: int,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUCheckpoint:
        self.check()

        pypicongpu_checkpoint = PyPIConGPUCheckpoint()
        pypicongpu_checkpoint.period = (
            self.period.get_as_pypicongpu(time_step_size, num_steps) if self.period is not None else None
        )
        pypicongpu_checkpoint.timePeriod = self.timePeriod
        pypicongpu_checkpoint.directory = self.directory
        pypicongpu_checkpoint.file = self.file
        pypicongpu_checkpoint.restart = self.restart
        pypicongpu_checkpoint.tryRestart = self.tryRestart
        pypicongpu_checkpoint.restartStep = self.restartStep
        pypicongpu_checkpoint.restartDirectory = self.restartDirectory
        pypicongpu_checkpoint.restartFile = self.restartFile
        pypicongpu_checkpoint.restartChunkSize = self.restartChunkSize
        pypicongpu_checkpoint.restartLoop = self.restartLoop
        pypicongpu_checkpoint.openPMD = self.openPMD

        return pypicongpu_checkpoint
