"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from typing import List
from ..rendering.renderedobject import RenderedObject
from ..util import build_typesafe_property

import typeguard


class _RangeSpecMeta(type):
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args)


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
class RangeSpec(RenderedObject, metaclass=_RangeSpecMeta):
    ranges = build_typesafe_property(List[slice])

    def __init__(self, *args):
        if not args:
            raise ValueError("RangeSpec must have at least one range")
        if len(args) > 3:
            raise ValueError(f"RangeSpec must have at most 3 ranges, got {len(args)}")
        if not all(isinstance(s, slice) for s in args):
            raise TypeError("All elements must be slice objects")
        for i, s in enumerate(args):
            if s.step is not None:
                raise ValueError(f"Step must be None in dimension {i + 1}, got {s.step}")
            if s.start is not None and not isinstance(s.start, int):
                raise TypeError(f"Begin in dimension {i + 1} must be int or None, got {type(s.start)}")
            if s.stop is not None and not isinstance(s.stop, int):
                raise TypeError(f"End in dimension {i + 1} must be int or None, got {type(s.stop)}")
        self.ranges = list(args)

    def _get_serialized(self) -> dict:
        return {"ranges": list(map(_serialize, self.ranges))}
