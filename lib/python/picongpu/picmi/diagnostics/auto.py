"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Pawel Ordyna, Masoud Afshari
License: GPLv3+
"""


from ...pypicongpu.output.auto import Auto as PyPIConGPUAuto
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies

from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard


@typeguard.typechecked
class Auto:
    """
    Specifies the parameters for the Auto output.

    Parameters
    ----------
    period: int
        Number of simulation steps between consecutive outputs.
        Unit: steps (simulation time steps).
    """

    period: TimeStepSpec
    """Number of simulation steps between consecutive outputs. Unit: steps (simulation time steps)."""

    def __init__(self, period: TimeStepSpec) -> None:
        self.period = period

    def check(self):
        pass

    def get_as_pypicongpu(
        self,
        # not used here, but needed for the interface
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUAuto:
        self.check()
        pypicongpu_auto = PyPIConGPUAuto()
        pypicongpu_auto.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        return pypicongpu_auto
