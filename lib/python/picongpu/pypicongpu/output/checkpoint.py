"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .timestepspec import TimeStepSpec


class Checkpoint(BaseModel):
    period: TimeStepSpec | None = None
    timePeriod: int | None = Field(default=None, ge=0)
    directory: Path | None = None
    file: str | None = None
    restart: bool | None = None
    tryRestart: bool | None = None
    restartStep: int | None = Field(default=None, ge=0)
    restartDirectory: str | None = None
    restartFile: str | None = None
    restartChunkSize: int | None = Field(default=None, gt=0)
    restartLoop: int | None = Field(default=None, ge=0)
    openPMD: dict | None = None

    type_checkpoint: Literal[True] = True

    @model_validator(mode="after")
    def check(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        return self
