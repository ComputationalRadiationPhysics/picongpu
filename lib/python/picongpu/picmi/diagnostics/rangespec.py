"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from typing import Tuple
from ...pypicongpu.output import RangeSpec as PyPIConGPURangeSpec
import re


class RangeSpec:
    """
    A class to specify a contiguous range of cells for simulation output in 1D, 2D, or 3D.

    This class parses a string in the format "begin:end" (1D), "begin:end,begin:end" (2D),
    or "begin:end,begin:end,begin:end" (3D) to define inclusive cell ranges for the simulation
    dimensions. For example:
        - 1D: RangeSpec("0:10") specifies cells 0 to 10 (x).
        - 2D: RangeSpec("0:10,5:15") specifies cells 0 to 10 (x) and 5 to 15 (y).
        - 3D: RangeSpec("0:10,5:15,2:8") specifies cells 0 to 10 (x), 5 to 15 (y), 2 to 8 (z).
    The default ":", ":,:" or ":,:,:," includes all cells in the simulation box for 1D, 2D or 3D.
    Values are clipped to the simulation box boundaries, and omitted bounds (":") indicate the
    full extent of the dimension. Negative indices are supported, counting from the end of the
    simulation box.

    Example usage:
        # 1D: rs = RangeSpec("0:10")  # x: cells 0 to 10
        #     rs = RangeSpec("-5:-1")  # x: last 5 cells (15 to 19 for 20-cell box)
        # 2D: rs = RangeSpec("0:10,5:15")  # x: 0 to 10, y: 5 to 15
        #     rs = RangeSpec("-5:-1,0:15")  # x: last 5 cells (15 to 19 for 20-cell box), y: 0 to 15
        # 3D: rs = RangeSpec("0:10,5:15,:")  # x: 0 to 10, y: 5 to 15, z: full range
        #     rs = RangeSpec("-5:-1,-10:-2,2:8")  # x: last 5 cells (15 to 19 for 20-cell box), y: last 9 to 2 cells (20 to 28 for 30-cell box), z: 2 to 8
    """

    def __init__(self, range_str: str):
        """
        Initialize a RangeSpec from a string.

        :param range_str: A string specifying cell ranges for 1 to 3 dimensions, e.g., "0:10"
                          (1D), "0:10,5:15" (2D), or "0:10,5:15,2:8" (3D).
        :raises ValueError: If the string format is invalid or contains non-integer bounds.
        """
        self.range_str = range_str
        self.slices = self._parse_range(range_str)
        self._validate()

    def _parse_range(self, range_str: str) -> Tuple[slice, ...]:
        """
        Parse the range string into a tuple of slice objects for each dimension.

        :param range_str: Input string (e.g., "0:10,5:15,2:8").
        :return: Tuple of 1 to 3 slice objects.
        :raises ValueError: If the string format is invalid or has more than 3 dimensions.
        """
        # Split the string into dimension parts
        parts = range_str.split(",")
        if len(parts) > 3:
            raise ValueError(f"Range must specify at most three dimensions, got {len(parts)}: {range_str}")

        slices = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part == ":":
                slices.append(slice(None, None, 1))
                continue

            # Match "begin:end" or single ":" using regex
            match = re.match(r"^([-]?\d+)?(:([-]?\d+)?)?$", part)
            if not match:
                raise ValueError(f"Invalid range format for dimension {i+1}: {part}. Expected 'begin:end' or ':'")

            start, _, end = match.groups()
            start = int(start) if start is not None else None
            end = int(end) if end is not None else None

            # Step is always 1 for contiguous ranges
            slices.append(slice(start, end, 1))

        return tuple(slices)

    def _validate(self):
        """
        Validate the parsed slices.

        :raises ValueError: If slices have invalid step values.
        """
        for i, s in enumerate(self.slices):
            if s.step is not None and s.step != 1:
                raise ValueError(
                    f"Step size must be 1 in dimension {i+1} since RangeSpec only supports contiguous ranges. Got {s.step}."
                )

    def _interpret_nones(self, spec: slice, dim_size: int) -> slice:
        """
        Replace None in slice bounds with simulation box limits (0 for start, dim_size-1 for stop).

        :param spec: Input slice.
        :param dim_size: Size of the simulation box in the dimension.
        :return: Slice with explicit bounds.
        """
        return slice(
            0 if spec.start is None else spec.start,
            dim_size - 1 if spec.stop is None else spec.stop,
            1,
        )

    def _interpret_negatives(self, spec: slice, dim_size: int) -> slice:
        """
        Convert negative indices to positive, clipping to simulation box.

        :param spec: Input slice.
        :param dim_size: Size of the simulation box in the dimension.
        :return: Slice with non-negative bounds, clipped to [0, dim_size-1].
        """
        if dim_size <= 0:
            raise ValueError(f"Dimension size must be positive. Got {dim_size}.")

        start = spec.start if spec.start is not None else 0
        stop = spec.stop if spec.stop is not None else dim_size - 1

        # Convert negative indices
        start = dim_size + start if start < 0 else start
        stop = dim_size + stop if stop < 0 else stop

        # Clip to simulation box
        start = max(0, min(start, dim_size - 1))
        stop = max(0, min(stop, dim_size - 1))

        # Ensure start <= stop for a valid range
        if start > stop:
            start, stop = stop, start

        return slice(start, stop, 1)

    def get_as_pypicongpu(self, simulation_box: Tuple[int, ...]) -> PyPIConGPURangeSpec:
        """
        Convert to a PyPIConGPURangeSpec object, applying simulation box clipping.

        :param simulation_box: Tuple of dimension sizes (1 to 3 dimensions).
        :return: PyPIConGPURangeSpec object with clipped, non-negative slices.
        :raises ValueError: If the number of ranges does not match the simulation box dimensions.
        """
        if len(self.slices) != len(simulation_box):
            raise ValueError(
                f"Number of range specifications ({len(self.slices)}) must match "
                f"simulation box dimensions ({len(simulation_box)})."
            )

        # Process each dimension
        processed_slices = [
            self._interpret_negatives(self._interpret_nones(s, dim_size), dim_size)
            for s, dim_size in zip(self.slices, simulation_box)
        ]

        return PyPIConGPURangeSpec(processed_slices)
