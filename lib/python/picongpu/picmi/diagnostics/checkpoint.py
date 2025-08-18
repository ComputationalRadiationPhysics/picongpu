"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.checkpoint import Checkpoint as PyPIConGPUCheckpoint
from .timestepspec import TimeStepSpec
import typeguard
from typing import Optional, Union, Dict


@typeguard.typechecked
class Checkpoint:
    """
    Specifies the parameters for creating checkpoints in PIConGPU simulations.

    Checkpoints allow saving the simulation state for restarting or analysis.

    Parameters
    ----------
    period: int or TimeStepSpec, optional
        Number of simulation steps between consecutive checkpoints (e.g., 10 for every 10 steps).
        Use 0 to disable checkpointing.
        Alternatively, a TimeStepSpec can be provided for specific step selection
        (e.g., TimeStepSpec([5, 10]), TimeStepSpec([slice(-10, None, 1)])).
        Unit: steps or seconds (via TimeStepSpec unit).
    timePeriod: int, optional
        Time interval between checkpoints in simulation time steps.
        Use 0 or None to disable time-based checkpointing.
    directory: str, optional
        Directory to store checkpoint files. Default: "checkpoints".
    file: str, optional
        Base name for checkpoint files. Default: None.
    restart: bool, optional
        Enable restarting from checkpoints. Default: True.
    tryRestart: bool, optional
        Attempt to restart from existing checkpoints. Default: False.
    restartStep: int, optional
        Specific step to restart from. Default: None.
    restartDirectory: str, optional
        Directory to look for restart files. Default: None.
    restartFile: str, optional
        Specific file to restart from. Default: None.
    restartChunkSize: int, optional
        Chunk size for reading restart data. Default: None.
    restartLoop: int, optional
        Number of restart loops. Default: None.
    openPMD: dict, optional
        Configuration for openPMD output (e.g., {"ext": "h5"}). Default: None.
    """

    def check(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided to enable checkpointing")
        if self.timePeriod is not None and (not isinstance(self.timePeriod, int) or self.timePeriod < 0):
            raise ValueError("timePeriod must be a non-negative integer")
        if self.restartStep is not None and self.restartStep < 0:
            raise ValueError("restartStep must be non-negative")
        if self.restartChunkSize is not None and self.restartChunkSize <= 0:
            raise ValueError("restartChunkSize must be positive")
        if self.restartLoop is not None and self.restartLoop < 0:
            raise ValueError("restartLoop must be non-negative")
        if (
            self.period is not None
            and isinstance(self.period, TimeStepSpec)
            and not self.period.get_as_pypicongpu(1.0, 200).get_rendering_context().get("specs", [])
            and (self.timePeriod is None or self.timePeriod == 0)
        ):
            import warnings

            warnings.warn(
                "Checkpoint is disabled because period is set to 0 or an empty TimeStepSpec and timePeriod is None or 0"
            )

    def __init__(
        self,
        period: Optional[Union[int, TimeStepSpec]] = None,
        timePeriod: Optional[int] = None,
        directory: Optional[str] = "checkpoints",
        file: Optional[str] = None,
        restart: Optional[bool] = True,
        tryRestart: Optional[bool] = False,
        restartStep: Optional[int] = None,
        restartDirectory: Optional[str] = None,
        restartFile: Optional[str] = None,
        restartChunkSize: Optional[int] = None,
        restartLoop: Optional[int] = None,
        openPMD: Optional[Dict] = None,
    ):
        if period is not None and not isinstance(period, (int, TimeStepSpec)):
            raise TypeError("period must be an integer or TimeStepSpec")
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec([slice(None, None, period)]) if period > 0 else TimeStepSpec()
        else:
            self.period = period if period is not None else TimeStepSpec()
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
        self.check()

    def get_as_pypicongpu(
        self,
        time_step_size: float,
        num_steps: int,
        species_map: Dict = {},
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUCheckpoint:
        self.check()
        pypicongpu_checkpoint = PyPIConGPUCheckpoint()
        pypicongpu_checkpoint.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
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
        pypicongpu_checkpoint._name = "checkpoint"
        return pypicongpu_checkpoint
