"""
This file is part of PIConGPU.
Copyright 2025-2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec


class FieldEnergyMonitor(BaseModel):
    period: TimeStepSpec
    type_fieldenergymonitor: Literal[True] = True
