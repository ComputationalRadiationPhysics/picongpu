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
import warnings


class Auto:
    """
    Specifies the parameters for the Auto output.

    Parameters
    ----------
    period: int or TimeStepSpec
        Number of simulation steps between consecutive outputs (e.g., 10 for every 10 steps).
        Use 0 to disable output.
        Alternatively, a TimeStepSpec can be provided for PIConGPU-specific step selection
        (e.g., TimeStepSpec[5, 10], TimeStepSpec[-10:]).
        Unit: steps (simulation time steps).
    """

    def __init__(self, period: int | TimeStepSpec) -> None:
        if not isinstance(period, (int, TimeStepSpec)):
            raise TypeError("period must be an integer or TimeStepSpec")
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec[::period]("steps") if period > 0 else TimeStepSpec()("steps")
        else:
            self.period = period
            if self.period.unit_system is None:
                self.period = self.period("steps")

    def check(self):
        if not self.period.get_as_pypicongpu(1.0, 100).get_rendering_context().get("specs", []):
            warnings.warn("Auto output is disabled because period is set to 0 or an empty TimeStepSpec")

    @typeguard.typechecked
    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUAuto:
        self.check()
        pypicongpu_auto = PyPIConGPUAuto()
        pypicongpu_auto.period = self.period.get_as_pypicongpu(time_step_size, num_steps)

        return pypicongpu_auto
