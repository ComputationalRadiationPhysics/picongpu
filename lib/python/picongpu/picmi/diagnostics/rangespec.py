"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.rangespec import RangeSpec as PyPIConGPURangeSpec
import re


class _RangeSpecMeta(type):
    """
    Custom metaclass providing the [] operator for RangeSpec.
    """

    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args)


class RangeSpec(metaclass=_RangeSpecMeta):
    """
    A class to specify a contiguous range of cells for simulation output in 1D, 2D, or 3D.

    This class stores a list of slices representing inclusive cell ranges for each dimension.
    Slices must have step=None (contiguous ranges) and integer or None endpoints. Use the []
    operator for concise syntax, e.g., RangeSpec[0:10, 5:15]. For example:
        - 1D: RangeSpec[0:10] specifies cells 0 to 10 (x).
        - 2D: RangeSpec[0:10, 5:15] specifies cells 0 to 10 (x), 5 to 15 (y).
        - 3D: RangeSpec[0:10, 5:15, 2:8] specifies cells 0 to 10 (x), 5 to 15 (y), 2 to 8 (z).
    The default RangeSpec[:] includes all cells in the simulation box for 1D.
    """

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

    @classmethod
    def from_string(cls, range_str: str) -> "RangeSpec":
        """
        Create a RangeSpec from a string (e.g., "0:10", "0:10,5:15").

        :param range_str: A string specifying cell ranges for 1 to 3 dimensions.
        :return: RangeSpec instance with parsed ranges.
        :raises TypeError: If range_str is not a string.
        :raises ValueError: If the string format is invalid or contains non-integer bounds.
        """
        if not isinstance(range_str, str):
            raise TypeError(f"range_str must be a string, got {type(range_str)}")
        if not range_str.strip():
            raise ValueError("range_str cannot be empty")

        # Split the string into dimension parts
        parts = range_str.split(",")
        if len(parts) > 3:
            raise ValueError(f"Range must specify at most three dimensions, got {len(parts)}")

        ranges = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part == ":":
                ranges.append(slice(None, None, None))
                continue

            # Match "begin:end" using regex, allowing negative integers
            match = re.match(r"^([-]?\d+)?(:([-]?\d+)?)?$", part)
            if not match:
                raise ValueError(f"Invalid range format for dimension {i+1}: {part}. Expected 'begin:end' or ':'")

            begin, _, end = match.groups()
            begin = int(begin) if begin is not None else None
            end = int(end) if end is not None else None
            ranges.append(slice(begin, end, None))

        return cls(*ranges)

    def _interpret_nones(self, spec: slice, dim_size: int) -> slice:
        """
        Replace None in slice bounds with simulation box limits (0 for begin, dim_size-1 for end).

        :param spec: Input slice.
        :param dim_size: Size of the simulation box in the dimension.
        :return: Slice with explicit bounds.
        """
        return slice(
            0 if spec.start is None else spec.start,
            dim_size - 1 if spec.stop is None else spec.stop,
            None,
        )

    def _interpret_negatives(self, spec: slice, dim_size: int) -> slice:
        """
        Convert negative indices to positive, clipping to simulation box.

        :param spec: Input slice.
        :param dim_size: Size of the simulation box in the dimension.
        :return: Slice with non-negative bounds, clipped to [0, dim_size-1].
        """
        if dim_size <= 0:
            raise ValueError(f"Dimension size must be positive. Got {dim_size}")

        begin = spec.start if spec.start is not None else 0
        end = spec.stop if spec.stop is not None else dim_size - 1

        # Convert negative indices
        begin = dim_size + begin if begin < 0 else begin
        end = dim_size + end if end < 0 else end

        # Clip to simulation box
        begin = max(0, min(begin, dim_size - 1))
        end = max(0, min(end, dim_size - 1))

        # Ensure begin <= end for a valid range
        if begin > end:
            begin, end = end, begin

        return slice(begin, end, None)

    def get_as_pypicongpu(self, simulation_box: tuple[int, ...]) -> PyPIConGPURangeSpec:
        """
        Convert to a PyPIConGPURangeSpec object, applying simulation box clipping.

        :param simulation_box: tuple of dimension sizes (1 to 3 dimensions).
        :return: PyPIConGPURangeSpec object with clipped, non-negative ranges.
        :raises ValueError: If the number of ranges does not match the simulation box dimensions.
        """
        if len(self.ranges) != len(simulation_box):
            raise ValueError(
                f"Number of range specifications ({len(self.ranges)}) must match "
                f"simulation box dimensions ({len(simulation_box)})"
            )

        # Process each dimension
        processed_ranges = [
            self._interpret_negatives(self._interpret_nones(s, dim_size), dim_size)
            for s, dim_size in zip(self.ranges, simulation_box)
        ]

        return PyPIConGPURangeSpec(*processed_ranges)
