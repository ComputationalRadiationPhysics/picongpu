"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from enum import Enum, EnumMeta
from math import ceil, floor
from ...pypicongpu.output import TimeStepSpec as PyPIConGPUTimeStepSpec
import typeguard


class CustomStrEnumMeta(EnumMeta):
    """
    Provides StrEnum-like functionality for Python < 3.12.
    """

    def __contains__(cls, val):
        try:
            cls(val)
        except ValueError:
            return False
        else:
            return True


class TimeStepUnits(Enum, metaclass=CustomStrEnumMeta):
    """
    Units allowed in TimeStepSpec.
    """

    STEPS = "steps"
    SECONDS = "seconds"

    @classmethod
    def _missing_(cls, value):
        value = str(value).lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError("Unknown unit in TimeStepSpec")


class _TimeStepSpecMeta(type):
    """
    Custom metaclass providing the [] operator for TimeStepSpec.
    """

    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args)


@typeguard.typechecked
class TimeStepSpec(metaclass=_TimeStepSpecMeta):
    """
    Specify time steps for simulation output using slices or indices.

    Use as: TimeStepSpec[:12:2, 7]("steps") or TimeStepSpec[1e-15:5e-15:2e-16]("seconds").
    Supports negative indices, inclusive slices, and addition of TimeStepSpec objects.

    Defaults to 'steps' unit unless explicitly set to 'seconds' via __call__('seconds').

    Examples for how TimeStepSpec is interpreted:

    Specific steps:
        TimeStepSpec([5, 10])
         [slice(5, 6, 1), slice(10, 11, 1)]
         steps: 5, 10

    Uniform interval:
        TimeStepSpec([slice(0, 100, 10)])
         steps: 0, 10, 20, ..., 90

    Infinite step range:
        TimeStepSpec([slice(0, None, 5)])
         steps: 0, 5, 10, 15, ...

    Negative start or stop:
        TimeStepSpec([slice(-10, -1, 1)])
         steps: -10, -9, ..., -2

    Mixed entries:
        TimeStepSpec([5, slice(20, 25, 2)])
         steps: 5, 20, 22, 24
    """

    def __init__(self, *args, specs_in_seconds=tuple()):
        self.specs = tuple()
        self.specs_in_seconds = tuple()
        self.unit_system = "steps"  # Default to steps

        if len(args) == 1 and isinstance(args[0], TimeStepSpec):
            self.specs = args[0].specs
            self.specs_in_seconds = args[0].specs_in_seconds
            self.unit_system = args[0].unit_system
            return

        if len(args) == 1 and isinstance(args[0], list):
            args = tuple(args[0])

        for spec in args:
            if not isinstance(spec, (slice, int, float)):
                raise TypeError(f"Invalid spec type: {type(spec)}")

        self.specs = tuple(spec if isinstance(spec, slice) else slice(spec, spec + 1, 1) for spec in args)
        self.specs_in_seconds = tuple(
            spec if isinstance(spec, slice) else slice(spec, spec + 1, 1) for spec in specs_in_seconds
        )

    def __call__(self, unit_system="steps"):
        if unit_system not in TimeStepUnits:
            raise ValueError("Unknown unit in TimeStepSpec")
        if self.unit_system != "steps" and self.unit_system != unit_system:
            raise ValueError(f"Cannot reset unit to {unit_system}, already set to {self.unit_system}")
        self.unit_system = unit_system
        if unit_system == "seconds":
            self.specs_in_seconds = self.specs
            self.specs = tuple()
        return self

    def __add__(self, other: "TimeStepSpec") -> "TimeStepSpec":
        if not isinstance(other, TimeStepSpec):
            raise TypeError(f"unsupported operand type(s) for +: TimeStepSpec and {type(other)}")
        if self.unit_system != other.unit_system and self.unit_system is not None and other.unit_system is not None:
            raise ValueError("Cannot add TimeStepSpec objects with different units")
        ts = TimeStepSpec(
            *self.specs,
            *other.specs,
            specs_in_seconds=(*self.specs_in_seconds, *other.specs_in_seconds),
        )
        ts.unit_system = self.unit_system or other.unit_system
        return ts

    def check(self):
        """
        Validate TimeStepSpec parameters.

        Raises
        ------
        ValueError
            If any step size is less than 1 (for steps) or less than 0 (for seconds).
        """
        specs = self.specs if self.unit_system == "steps" else self.specs_in_seconds
        for spec in specs:
            step = spec.step if isinstance(spec, slice) else 1
            if step is not None:
                if self.unit_system == "steps" and step < 1:
                    raise ValueError(f"Step size must be >= 1 in TimeStepSpec. You gave {step}.")
                if self.unit_system == "seconds" and step <= 0:
                    raise ValueError(f"Step size must be > 0 in TimeStepSpec. You gave {step}.")

    def _interpret_nones(self, spec, num_steps):
        """
        Replace None in slice bounds with simulation limits (0 for start, num_steps for stop).
        """
        return slice(
            0 if spec.start is None else spec.start,
            -1 if spec.stop is None else spec.stop,
            spec.step if spec.step is not None else 1,
        )

    def _interpret_negatives(self, spec, num_steps):
        step = spec.step if spec.step is not None else 1
        if self.unit_system == "steps" and step < 1:
            raise ValueError(f"Step size must be >= 1 in TimeStepSpec. You gave {step}.")

        start = spec.start if spec.start is None or spec.start >= 0 else num_steps + spec.start

        # Only convert stop if it's not None and not -1
        if spec.stop is None:
            stop = -1
        elif spec.stop < 0:
            stop = num_steps + spec.stop
        else:
            stop = spec.stop

        if stop == -1:
            stop = num_steps
        else:
            stop = max(start, min(stop, num_steps))

        return slice(start, stop, step)

    def get_as_pypicongpu(self, time_step_size: float, num_steps: int) -> PyPIConGPUTimeStepSpec:
        """
        Convert to PyPIConGPU TimeStepSpec with resolved indices.

        :param time_step_size: Size of one time step in seconds (must be positive).
        :param num_steps: Total number of simulation steps (must be positive).
        :return: PyPIConGPUTimeStepSpec with clipped, inclusive ranges as slice objects.
        """
        if time_step_size <= 0:
            raise ValueError("time_step_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        specs = self.specs if self.unit_system in ["steps", None] else self.specs_in_seconds
        resolved_specs = []

        for spec in specs:
            # Handle single time points
            if not isinstance(spec, slice) or (
                spec.start is not None and spec.stop is not None and spec.start + 1 == spec.stop and spec.step == 1
            ):
                index = spec.start if isinstance(spec, slice) else spec
                if self.unit_system == "seconds":
                    index = floor(index / time_step_size)
                if index < 0:
                    index = max(0, num_steps + index)
                if index >= num_steps:
                    continue
                resolved_specs.append(slice(index, index + 1, 1))
                continue

            # Process slices
            spec = self._interpret_nones(spec, num_steps)
            spec = self._interpret_negatives(spec, num_steps)

            start = spec.start
            stop = spec.stop
            step = spec.step

            if self.unit_system == "seconds":
                start = floor(start / time_step_size)
                stop = ceil(stop / time_step_size)
                step = ceil(step / time_step_size)
                if step < 1:
                    raise ValueError(f"Step size must be >= 1 in TimeStepSpec. You gave {step}.")

            # Clip to valid range
            start = max(0, min(start, num_steps))

            if stop == -1:
                stop = num_steps
            else:
                stop = max(start, min(stop, num_steps))

            if start < stop:
                resolved_specs.append(slice(start, stop, step))

        return PyPIConGPUTimeStepSpec(specs=resolved_specs)
