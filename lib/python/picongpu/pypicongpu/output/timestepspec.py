"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from ..rendering.renderedobject import RenderedObject
from ..util import build_typesafe_property

import typeguard


def _serialize(spec):
    return {
        "start": spec.start if spec.start is not None else 0,
        "stop": spec.stop if spec.stop is not None else -1,
        "step": spec.step if spec.step is not None else 1,
    }


@typeguard.typechecked
class TimeStepSpec(RenderedObject):
    specs = build_typesafe_property(list[slice])

    def __init__(self, specs: list[slice]):
        self.specs = specs
        self.check()

    def check(self):
        for spec in self.specs:
            if spec.step is not None and spec.step < 1:
                raise ValueError("Step size must be >= 1")

    def get_as_pypicongpu(self, time_step_size: float, num_steps: int) -> "TimeStepSpec":
        """
        Convert to a pypicongpu TimeStepSpec object with resolved indices.

        :param time_step_size: Size of one time step in seconds (must be positive).
        :param num_steps: Total number of simulation steps (must be positive).
        :return: A new TimeStepSpec with resolved indices.
        """
        if time_step_size <= 0:
            raise ValueError("Time step size must be strictly positive")
        if num_steps <= 0:
            raise ValueError("Number of steps must be positive")

        self.check()
        resolved_specs = []
        for spec in self.specs:
            start = spec.start if spec.start is not None else 0
            stop = spec.stop if spec.stop is not None else num_steps - 1
            step = spec.step if spec.step is not None else 1

            # Resolve negative indices
            if start < 0:
                start = max(0, num_steps + start)
            if stop < 0:
                stop = max(0, num_steps + stop)

            # Ensure start <= stop
            if start > stop:
                start, stop = stop, start

            resolved_specs.append(slice(start, stop + 1, step))

        return TimeStepSpec(resolved_specs)

    def _get_serialized(self):
        self.check()
        return {"specs": list(map(_serialize, self.specs))}
