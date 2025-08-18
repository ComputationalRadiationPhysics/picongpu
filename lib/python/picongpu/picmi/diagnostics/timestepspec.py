"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import math
from typing import List, Union, Optional
from ...pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec
import typeguard


@typeguard.typechecked
class TimeStepSpec:
    """
    Defines specific simulation time steps for PIConGPU diagnostics.

    Allows specification of time steps for diagnostic output, either as individual steps,
    step ranges, or in seconds (converted to steps based on time step size).
    Supports negative indexing and addition of TimeStepSpec objects.

    Parameters
    ----------
    specs: List[Union[int, slice]], optional
        List of time step specifications. Each spec can be:
        - An integer for a specific step (e.g., 5 for step 5).
        - A slice for a range of steps (e.g., slice(0, 100, 10) for every 10 steps from 0 to 99).
        Default is an empty list, meaning no steps are selected.
    unit: str, optional
        Unit of the time steps, either "steps" or "seconds". Default is "steps".
    """

    def __init__(self, specs: Optional[List[Union[int, slice]]] = None, unit: str = "steps"):
        self.specs = specs if specs is not None else []
        if unit not in ["steps", "seconds"]:
            raise ValueError("Unit must be 'steps' or 'seconds'")
        self.unit = unit

    def __add__(self, other: "TimeStepSpec") -> "TimeStepSpec":
        """
        Combine two TimeStepSpec objects by merging their specs lists.

        Parameters
        ----------
        other: TimeStepSpec
            The other TimeStepSpec to add.

        Returns
        -------
        TimeStepSpec
            A new TimeStepSpec with combined specs, maintaining the unit of the first object.
        """
        if not isinstance(other, TimeStepSpec):
            raise TypeError("Can only add TimeStepSpec to another TimeStepSpec")
        if self.unit != other.unit:
            raise ValueError("Cannot add TimeStepSpec objects with different units")
        combined_specs = self.specs + other.specs
        return TimeStepSpec(combined_specs, self.unit)

    def get_as_pypicongpu(self, time_step_size: float, num_steps: int) -> PyPIConGPUTimeStepSpec:
        """
        Convert TimeStepSpec to PyPIConGPUTimeStepSpec for rendering.

        Parameters
        ----------
        time_step_size: float
            The size of a single time step in seconds (must be positive).
        num_steps: int
            The total number of simulation steps (must be positive).

        Returns
        -------
        PyPIConGPUTimeStepSpec
            The equivalent PyPIConGPUTimeStepSpec object with resolved time steps.
        """
        if time_step_size <= 0:
            raise ValueError("Time step size must be strictly positive")
        if num_steps <= 0:
            raise ValueError("Number of steps must be positive")

        resolved_specs = []
        for spec in self.specs:
            if isinstance(spec, int):
                step = spec
                if step < 0:
                    step = num_steps + step
                if 0 <= step < num_steps:
                    resolved_specs.append(slice(step, step + 1, 1))
            elif isinstance(spec, slice):
                start = spec.start if spec.start is not None else 0
                stop = spec.stop if spec.stop is not None else num_steps
                step = spec.step if spec.step is not None else 1
                if step <= 0:
                    raise ValueError("Step size must be >= 1")
                if start < 0:
                    start = num_steps + start
                if stop < 0:
                    stop = num_steps + stop
                if start < 0 or stop > num_steps or start >= num_steps:
                    continue
                if self.unit == "seconds":
                    start = math.floor(start / time_step_size)
                    stop = math.ceil(stop / time_step_size)
                resolved_specs.append(slice(start, stop, step))

        pypicongpu_spec = PyPIConGPUTimeStepSpec(specs=resolved_specs)
        return pypicongpu_spec
