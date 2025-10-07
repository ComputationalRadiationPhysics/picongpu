"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ..rendering.renderedobject import RenderedObject
from ..util import build_typesafe_property

import typeguard


def _serialize(spec: slice) -> dict:
    if not isinstance(spec, slice):
        raise ValueError(f"Expected a slice for range, got {type(spec)}")
    if spec.start is not None and not isinstance(spec.start, int):
        raise ValueError(f"Begin must be int or None, got {type(spec.start)}")
    if spec.stop is not None and not isinstance(spec.stop, int):
        raise ValueError(f"End must be int or None, got {type(spec.stop)}")
    return {
        "begin": spec.start if spec.start is not None else 0,
        "end": spec.stop if spec.stop is not None else -1,
    }


@typeguard.typechecked
class RangeSpec(RenderedObject):
    ranges = build_typesafe_property(list[slice])

    def __init__(self, ranges: list[slice]):
        self.ranges = ranges

    def _get_serialized(self) -> dict:
        return {"ranges": list(map(_serialize, self.ranges))}
