"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from picongpu.picmi.copy_attributes import default_converts_to

from ...pypicongpu.output.checkpoint import Checkpoint as PyPIConGPUCheckpoint
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUCheckpoint)
class Checkpoint(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    """
    Specifies the parameters for creating checkpoints in PIConGPU simulations.

    This plugin saves simulation state snapshots at specified intervals,
    allowing for simulation restarts or analysis.

    Attention: ** At least one of period or timePeriod must be provided.**

    Parameters
    ----------
    period: TimeStepSpec, optional
        Specify on which time steps to create checkpoints.
    timePeriod: int, optional
        Specify the interval in minutes for creating checkpoints.
    directory: str | Path, optional
        Directory inside simOutput for writing checkpoints.
    file: str, optional
        Relative or absolute fileset prefix for checkpoint files.
    restart: bool, optional
        If True, restart simulation from the latest checkpoint.
    tryRestart: bool, optional
        If True, restart from the latest checkpoint if available.
    restartStep: int, optional
        Specific checkpoint step to restart from.
    restartDirectory: str, optional
        Directory inside simOutput containing checkpoints for restart.
    restartFile: str, optional
        Relative or absolute fileset prefix for reading checkpoints.
    restartChunkSize: int, optional
        Number of particles processed per kernel call during restart.
    restartLoop: int, optional
        Number of times to restart the simulation after it finishes.
    openPMD: dict, optional
        Dictionary of openPMD-specific settings.
    """

    period: TimeStepSpec | None = None
    timePeriod: int | None = Field(default=None, ge=0)
    directory: str | Path | None = None
    file: str | None = None
    restart: bool | None = None
    tryRestart: bool | None = None
    restartStep: int | None = Field(default=None, ge=0)
    restartDirectory: str | None = None
    restartFile: str | None = None
    restartChunkSize: int | None = Field(default=None, gt=0)
    restartLoop: int | None = Field(default=None, ge=0)
    openPMD: dict | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        return self

    def check(self, *args, **kwargs):
        pass
