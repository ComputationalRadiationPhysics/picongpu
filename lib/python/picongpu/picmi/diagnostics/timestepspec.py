# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from enum import Enum, EnumMeta
from math import ceil

from ...pypicongpu.output import TimeStepSpec as PyPIConGPUTimeStepSpec


class CustomStrEnumMeta(EnumMeta):
    """
    This class provides some functionality of 3.12 StrEnum,
    namely its __contains__() method.

    You can safely remove this and inherit directly from StrEnum
    once we switched to 3.12.
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
        raise ValueError(f"Unknown time step unit. You gave {value}.")


# The following might look slightly convoluted but is preferred over the more obvious use of `__class_getitem__`.
# This is because the latter is supposed to return a GenericAlias which is not what we want.
# That would be like list[int].
# We are more like an `Enum` where indexing into the class means something different.
# See [here](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) and
# [here](https://docs.python.org/3/reference/datamodel.html#classgetitem-versus-getitem).


class _TimeStepSpecMeta(type):
    """
    Custom metaclass providing the [] operator on its children.
    """

    # Provide this to have a nice syntax picmi.diagnostics.TimeStepSpec[10:200:5, 3, 7, 11, 17]
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args)


class TimeStepSpec(metaclass=_TimeStepSpecMeta):
    """
    A class to specify time steps for simulation output.

    This class allows for flexible specification of time steps using slices
    or individual indices. Its custom metaclass provides a [] operator on the class itself
    for slicing and the () operator for choosing the unit such that the most convenient
    way to use it is as follows:

        ts = TimeStepSpec[:12:2, 7]("steps") + TimeStepSpec[1.e-15:5.e-15:2.e-16]("seconds")

    In this example, `ts` specifies:
    - every other time step for the first 12 time steps inclusively (`0, 2, 4, 6, 8, 10, 12`)
    - AND the 7th time step
    - AND one output every 2.e-16 seconds in the (inclusive) range 1.e-15 to 5.e-15 seconds
      (which indices this maps to depends on the time step size and the number of time steps)

    In general, the class implements the following semantics for the operator []:
    - the [] operator understands slices and numbers separated by commas
    - specifications separated by commas are interpreted as unions
    - slices are interpreted as inclusive on both ends, so `start:stop:step` includes the values
      `start` and `stop` (if there exists an integer n such that `n*step+start == stop`)
    - negative values are allowed for `start` and `stop` but not `step` (due to practical limitations
      of the simulation code); as expected in Python they count from the end but due to inclusiveness
      `:-1` includes the last element
    - individual numbers denote a single time step
    - multiple specifications (particularly in different units) can be concatenated (as set unions)
      via the + operator

    Default units are `steps`. If other units are given (see which are implemented in `TimeStepUnits`),
    rounding must happen in the translation into steps (the only unit available in the backend). This
    rounding is implemented to round down (up) for the lower (upper) bound such that the interval will
    never be clipped. The time step is always rounded down to the next available multiple of the time
    step size, such that for long and sparsely sampled intervals distortions may occur.

    An extensive list of tests is available in the corresponding directory, mapping the syntax to
    concrete index sets. The reader is encouraged to look for clarification there.
    """

    unit_system = None

    def __init__(self, *args, specs_in_seconds=tuple()):
        self.specs = tuple()
        self.specs_in_seconds = tuple()

        # allow copy initialisation from another TimeStepSpec.
        if len(args) == 1 and isinstance(args[0], TimeStepSpec):
            self.specs = args[0].specs
            self.specs_in_seconds = args[0].specs_in_seconds
            return

        self.specs = tuple(
            # The else branch is supposed to handle integers.
            # We use a slice here because PIConGPU's interpretation of the
            # --period argument for single integers is different.
            # In PIConGPU, a single integer would be interpreted as
            # slice(None, None, value) but this is unnatural for the
            # Python [] operator.
            spec if isinstance(spec, slice) else slice(spec, spec, None)
            for spec in args
        )
        self.specs_in_seconds = tuple(specs_in_seconds)

    def __call__(self, unit_system="steps"):
        if unit_system not in TimeStepUnits:
            raise ValueError(f"Unknown unit in TimeStepSpec. You gave {unit_system} which is not in TimeStepUnits.")
        if self.unit_system is not None and self.unit_system != unit_system:
            raise ValueError(
                "Don't reset units on a TimeStepSpec. "
                f"You've tried to set {unit_system} but it's already {self.unit_system}."
            )
        self.unit_system = unit_system
        if unit_system == "seconds":
            self.specs_in_seconds = self.specs
            self.specs = tuple()
        return self

    def __add__(self, other):
        if not (isinstance(other, TimeStepSpec)):
            raise TypeError(f"unsupported operand type(s) for +: TimeStepSpec and {type(other)}")
        ts = TimeStepSpec(
            *self.specs,
            *other.specs,
            specs_in_seconds=(*self.specs_in_seconds, *other.specs_in_seconds),
        )
        # The following guards against setting units on the result of the addition.
        # Otherwise one could specify time steps in "steps" unit, add that to
        # another TimeStepSpec and reset the units.
        ts.unit_system = "mixed"
        return ts

    def _transform_to_steps(self, specs_in_seconds, time_step_size):
        if time_step_size <= 0:
            raise ValueError(f"Time step size must be strictly positive. You gave {time_step_size}.")
        return tuple(
            slice(
                int(spec.start / time_step_size if spec.start is not None else 0),
                int(ceil(spec.stop / time_step_size)) if spec.stop is not None else None,
                int(spec.step / time_step_size if spec.step is not None else 1) or 1,
            )
            for spec in specs_in_seconds
        )

    def _interpret_nones(self, spec):
        # We must communicate an open end explicitly, so we leave spec.stop as None.
        return slice(
            spec.start if spec.start is not None else 0,
            spec.stop if spec.stop is not None else -1,
            spec.step if spec.step is not None else 1,
        )

    def _interpret_negatives(self, spec, num_steps):
        if spec.step < 1:
            raise ValueError(f"Step size must be >= 1 in TimeStepSpec. You gave {spec.step}.")
        return slice(
            spec.start if spec.start >= 0 else num_steps + spec.start,
            spec.stop if (spec.stop is None or spec.stop >= -1) else num_steps + spec.stop,
            spec.step,
        )

    def get_as_pypicongpu(self, time_step_size, num_steps, **kwargs):
        """
        Creates the corresponding pypicongpu object by translating every specification
        into non-negative (except for -1) slices in units of steps. It takes `time_step_size`
        and `num_steps` to compute this transformation.
        """
        return PyPIConGPUTimeStepSpec(
            [
                self._interpret_negatives(self._interpret_nones(s), num_steps)
                for s in self.specs + self._transform_to_steps(self.specs_in_seconds, time_step_size)
            ]
        )
