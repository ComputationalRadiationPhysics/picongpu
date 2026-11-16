"""
This file is part of PIConGPU.
Copyright 2025-2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picmistandard import PICMI_Diagnostic
from pydantic import ConfigDict, Field

from picongpu.picmi._copy_attributes import default_converts_to

from ...pypicongpu.output.field_energy_monitor import (
    FieldEnergyMonitor as PyPIConGPUFieldEnergyMonitor,
)
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUFieldEnergyMonitor)
class FieldEnergyMonitor(PICMI_Diagnostic):
    """
    Specifies the parameters for monitoring the total energy of the electromagnetic fields.

    This free (species-independent) plugin tracks the total field energy
    (energy stored in the electric and magnetic fields) over the course of the
    simulation, useful for verifying energy conservation.
    """

    period: TimeStepSpec = Field(description="Specify on which time steps to record the total field energy.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
