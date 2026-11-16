"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pathlib import Path

from picmistandard import PICMI_Diagnostic
from pydantic import Field, model_validator

from picongpu.picmi._copy_attributes import default_converts_to

from ...pypicongpu.output.checkpoint import Checkpoint as PyPIConGPUCheckpoint
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUCheckpoint)
class Checkpoint(PICMI_Diagnostic):
    """
    Specifies the parameters for creating checkpoints in PIConGPU simulations.

    This plugin saves simulation state snapshots at specified intervals,
    allowing for simulation restarts or analysis.

    Attention: ** At least one of period or timePeriod must be provided.**
    """

    model_config = {"arbitrary_types_allowed": True}

    period: TimeStepSpec | None = Field(default=None, description="Specify on which time steps to create checkpoints.")
    timePeriod: int | None = Field(
        default=None, ge=0, description="Specify the interval in minutes for creating checkpoints."
    )
    directory: str | Path | None = Field(
        default=None, description="Directory inside simOutput for writing checkpoints."
    )
    file: str | None = Field(default=None, description="Relative or absolute fileset prefix for checkpoint files.")
    restart: bool | None = Field(default=None, description="If True, restart simulation from the latest checkpoint.")
    tryRestart: bool | None = Field(
        default=None, description="If True, restart from the latest checkpoint if available."
    )
    restartStep: int | None = Field(default=None, ge=0, description="Specific checkpoint step to restart from.")
    restartDirectory: str | None = Field(
        default=None, description="Directory inside simOutput containing checkpoints for restart."
    )
    restartFile: str | None = Field(
        default=None, description="Relative or absolute fileset prefix for reading checkpoints."
    )
    restartChunkSize: int | None = Field(
        default=None, gt=0, description="Number of particles processed per kernel call during restart."
    )
    restartLoop: int | None = Field(
        default=None, ge=0, description="Number of times to restart the simulation after it finishes."
    )
    openPMD: dict | None = Field(default=None, description="Dictionary of openPMD-specific settings.")

    @model_validator(mode="after")
    def _validate(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        return self

    def check(self, *args, **kwargs):
        pass
