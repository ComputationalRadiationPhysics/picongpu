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
    """
    Custom metaclass providing the [] operator for RangeSpec.
    """

    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args)


def _serialize(spec: slice) -> dict:
    """
    Serialize a slice object to a JSON-compatible dictionary.

    :param spec: A slice object representing a range for one dimension.
    :return: Dictionary with begin and end keys.
    :raises ValueError: If the input is not a slice or has invalid endpoints.
    """
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
    """
    A class to specify a contiguous range of cells for simulation output in 1D, 2D, or 3D.

    This class stores a list of slices representing inclusive cell ranges for each dimension.
    It is used as the output of PICMI RangeSpec.get_as_pypicongpu, with negative indices and
    clipping already handled. Slices must have step=None (contiguous ranges) and integer or None
    endpoints. Use the [] operator for concise syntax, e.g., RangeSpec[0:10, 5:15].
    Example:
        - 1D: RangeSpec[0:10] specifies cells 0 to 10 (x).
        - 2D: RangeSpec[0:10, 5:15] specifies cells 0 to 10 (x), 5 to 15 (y).
        - 3D: RangeSpec[0:10, 5:15, 2:8] specifies cells 0 to 10 (x), 5 to 15 (y), 2 to 8 (z).
    """

    ranges = build_typesafe_property(List[slice])

    def __init__(self, *args):
        """
        Initialize a RangeSpec with a list of slices.

        :param args: 1 to 3 slice objects, e.g., slice(0, 10), slice(5, 15).
        :raises TypeError: If args contains non-slice elements or invalid endpoint types.
        :raises ValueError: If args is empty, has more than 3 slices, or contains slices with step != None.
        """
        if not args:
            raise ValueError("RangeSpec must have at least one range")
        if len(args) > 3:
            raise ValueError(f"RangeSpec must have at most 3 ranges, got {len(args)}")
        if not all(isinstance(s, slice) for s in args):
            raise TypeError("All elements must be slice objects")
        for i, s in enumerate(args):
            if s.step is not None:
                raise ValueError(f"Step must be None in dimension {i+1}, got {s.step}")
            if s.start is not None and not isinstance(s.start, int):
                raise TypeError(f"Begin in dimension {i+1} must be int or None, got {type(s.start)}")
            if s.stop is not None and not isinstance(s.stop, int):
                raise TypeError(f"End in dimension {i+1} must be int or None, got {type(s.stop)}")
        self.ranges = list(args)

    def _get_serialized(self) -> dict:
        """
        Serialize the RangeSpec to a JSON-compatible dictionary.

        :return: Dictionary with serialized ranges.
        """
        return {"ranges": list(map(_serialize, self.ranges))}
