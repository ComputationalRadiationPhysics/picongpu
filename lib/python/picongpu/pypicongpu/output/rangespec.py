"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ..rendering.renderedobject import RenderedObject
from ..util import build_typesafe_property

import typeguard


def _serialize(spec):
    """
    Serialize a slice object to a JSON-compatible dictionary.

    :param spec: A slice object representing a range for one dimension.
    :return: Dictionary with start, stop, and step keys.
    :raises ValueError: If the input is not a slice.
    """
    if isinstance(spec, slice):
        return {
            "start": spec.start if spec.start is not None else 0,
            "stop": spec.stop if spec.stop is not None else -1,
            "step": spec.step if spec.step is not None else 1,
        }
    raise ValueError(f"Unknown serialization for {spec=} as a range specifier.")


@typeguard.typechecked
class RangeSpec(RenderedObject):
    """
    A class to specify a contiguous range of cells for simulation output in 1D, 2D, or 3D.

    This class stores a list of slices representing inclusive cell ranges for each dimension.
    It is used as the output of PICMI RangeSpec.get_as_pypicongpu, with negative indices and
    clipping already handled. The slices must have a step size of 1 (contiguous ranges).

    """

    specs = build_typesafe_property(list[slice])

    def __init__(self, specs: list[slice]):
        """
        Initialize a RangeSpec with a list of slices.

        :param specs: List of 1 to 3 slice objects, one per dimension.
        :raises ValueError: If specs is empty, has more than 3 slices, or contains slices with step != 1.
        """
        if not specs:
            raise ValueError("RangeSpec must have at least one slice.")
        if len(specs) > 3:
            raise ValueError(f"RangeSpec must have at most 3 slices, got {len(specs)}.")
        for i, spec in enumerate(specs):
            if spec.step is not None and spec.step != 1:
                raise ValueError(
                    f"Step size must be 1 in dimension {i+1} since RangeSpec only supports contiguous ranges. Got {spec.step}."
                )
        self.specs = specs

    def _get_serialized(self):
        return {"specs": list(map(_serialize, self.specs))}
