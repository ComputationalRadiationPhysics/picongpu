"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""
import typeguard
from ..rendering.renderedobject import RenderedObject
from ..util import build_typesafe_property


def _serialize(spec):
    if isinstance(spec, slice):
        return {
            "start": spec.start if spec.start is not None else 0,
            "stop": spec.stop if spec.stop is not None else -1,
            "step": spec.step if spec.step is not None else 1,
        }
    raise ValueError(f"Unknown serialization for {spec=} as a time step specifier (--period argument).")


@typeguard.typechecked
class TimeStepSpec(RenderedObject):
    specs = build_typesafe_property(list[slice])

    def __init__(self, specs: list[slice]):
        # Here, you could add normalization checks if you want (optional)
        self.specs = specs

    def check(self):
        """
        Validate the TimeStepSpec.

        Ensures all slices have positive step sizes and valid slice types.
        """
        for spec in self.specs:
            if not isinstance(spec, slice):
                raise ValueError(f"Expected slice, got {type(spec)}")
            if spec.step is not None and spec.step < 1:
                raise ValueError("Step size must be >= 1")

    def _get_serialized(self):
        return {"specs": list(map(_serialize, self.specs))}
