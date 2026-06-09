# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from pathlib import Path
from typing import Annotated, Literal

import typeguard
from pydantic import BaseModel, Field, model_validator

from .timestepspec import TimeStepSpec


@typeguard.typechecked
class Checkpoint(BaseModel):
    period: TimeStepSpec | None
    timePeriod: Annotated[int, Field(..., ge=0)] | None
    directory: Path | None
    file: str | None
    restart: bool | None
    tryRestart: bool | None
    restartStep: Annotated[int, Field(..., ge=0)] | None
    restartDirectory: str | None
    restartFile: str | None
    restartChunkSize: Annotated[int, Field(..., gt=0)] | None
    restartLoop: Annotated[int, Field(..., ge=0)] | None
    openPMD: dict | None

    type_checkpoint: Literal[True] = True

    @model_validator(mode="after")
    def check(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        return self
