"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from enum import StrEnum
from math import ceil, floor

from ...pypicongpu.output import TimeStepSpec as PyPIConGPUTimeStepSpec


class TimeStepUnits(StrEnum):
    """
    Units allowed in TimeStepSpec.

    This is a plain :class:`enum.StrEnum` (Python >= 3.11), so the former
    custom metaclass that emulated ``StrEnum`` on older Python versions is
    gone. Unit names are still accepted case-insensitively via ``_missing_``.
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


class TimeStepShift:
    """
    A pending shift of a :class:`TimeStepSpec` by ``amount`` units.

    Instances are produced by multiplying one of the unit accessors of
    :class:`TimeStepSpec`, for instance ``10 * TimeStepSpec.steps`` or
    ``2.e-6 * TimeStepSpec.seconds``, and are added to a ``TimeStepSpec``.
    The shift is deliberately *not* resolved at creation time: a shift may be
    given in seconds while the simulation time step size is only known at
    translation time, so it is stored on the spec and applied in
    :meth:`TimeStepSpec.get_as_pypicongpu`.
    """

    def __init__(self, amount, unit):
        self.amount = amount
        self.unit = TimeStepUnits(unit)

    def __rmul__(self, factor):
        return TimeStepShift(self.amount * factor, self.unit)

    def __mul__(self, factor):
        return TimeStepShift(self.amount * factor, self.unit)

    def __eq__(self, other):
        if not isinstance(other, TimeStepShift):
            return NotImplemented
        return self.amount == other.amount and self.unit == other.unit

    def __hash__(self):
        return hash((self.amount, self.unit))

    def __repr__(self):
        return f"TimeStepShift({self.amount!r}, {self.unit.value!r})"


# The following might look slightly convoluted but is preferred over the more obvious use of `__class_getitem__`.
# This is because the latter is supposed to return a GenericAlias which is not what we want.
# That would be like list[int].
# We are more like an `Enum` where indexing into the class means something different.
# See [here](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) and
# [here](https://docs.python.org/3/reference/datamodel.html#classgetitem-versus-getitem).


class _TimeStepSpecMeta(type):
    """
    Custom metaclass providing the [] operator and the unit accessors on its children.
    """

    # Provide this to have a nice syntax picmi.diagnostics.TimeStepSpec[10:200:5, 3, 7, 11, 17]
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        return cls(*args)

    @property
    def steps(cls):
        """
        Unit accessor for a shift measured in simulation steps.

        ``10 * TimeStepSpec.steps`` is a shift by ten steps, to be added to a
        :class:`TimeStepSpec`.
        """
        return TimeStepShift(1, TimeStepUnits.STEPS)

    @property
    def seconds(cls):
        """
        Unit accessor for a shift measured in physical time.

        ``2.e-6 * TimeStepSpec.seconds`` is a shift by two microseconds, to be
        added to a :class:`TimeStepSpec`.
        """
        return TimeStepShift(1, TimeStepUnits.SECONDS)


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

    A whole specification (or any of its parts) may additionally be shifted by a number of
    steps or a physical time via the unit accessors `TimeStepSpec.steps` and
    `TimeStepSpec.seconds`, for instance

        ts = (TimeStepSpec[::1]("steps") + 10 * TimeStepSpec.steps) + (
            TimeStepSpec[::1.e-5] + 2.e-6 * TimeStepSpec.seconds
        )

    The shift is applied to every `start`/`stop` of the specification while leaving the
    `step` untouched; open ends stay open. Shifts are resolved during the translation to
    the backend (in `get_as_pypicongpu`), where the simulation time step size is known, so
    a shift may be written in seconds even for a spec that will only be scaled later. An
    unqualified spec adopts the unit of the first shift applied to it, so the example above
    yields a steps-unit spec shifted by ten steps unioned with a seconds-unit spec shifted
    by two microseconds.

    An extensive list of tests is available in the corresponding directory, mapping the syntax to
    concrete index sets. The reader is encouraged to look for clarification there.
    """

    unit_system = None

    def __init__(self, *args, specs_in_seconds=tuple(), shifts=tuple(), shifts_in_seconds=tuple()):
        self.specs = tuple()
        self.specs_in_seconds = tuple()
        self.shifts = tuple()
        self.shifts_in_seconds = tuple()

        # allow copy initialisation from another TimeStepSpec.
        if len(args) == 1 and isinstance(args[0], TimeStepSpec):
            other = args[0]
            self.specs = other.specs
            self.specs_in_seconds = other.specs_in_seconds
            self.shifts = other.shifts
            self.shifts_in_seconds = other.shifts_in_seconds
            self.unit_system = other.unit_system
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
        # `shifts` holds one (possibly empty) tuple of pending shifts per spec,
        # so a shift stays attached to the part it was added to when two specs
        # are unioned via `+`.
        self.shifts = tuple(shifts) if shifts else (tuple(),) * len(self.specs)
        self.shifts_in_seconds = (
            tuple(shifts_in_seconds) if shifts_in_seconds else (tuple(),) * len(self.specs_in_seconds)
        )

    def __call__(self, unit_system="steps"):
        try:
            unit = TimeStepUnits(unit_system)
        except ValueError:
            raise ValueError(
                f"Unknown unit in TimeStepSpec. You gave {unit_system} which is not in TimeStepUnits."
            ) from None
        if self.unit_system is not None and self.unit_system != unit.value:
            raise ValueError(
                "Don't reset units on a TimeStepSpec. "
                f"You've tried to set {unit.value} but it's already {self.unit_system}."
            )
        self.unit_system = unit.value
        if unit is TimeStepUnits.SECONDS:
            self.specs_in_seconds = (*self.specs_in_seconds, *self.specs)
            self.specs = tuple()
            self.shifts_in_seconds = (*self.shifts_in_seconds, *self.shifts)
            self.shifts = tuple()
        return self

    def __add__(self, other):
        if isinstance(other, TimeStepShift):
            return self._with_shift(other)
        if not (isinstance(other, TimeStepSpec)):
            raise TypeError(f"unsupported operand type(s) for +: TimeStepSpec and {type(other)}")
        ts = TimeStepSpec(
            *self.specs,
            *other.specs,
            specs_in_seconds=(*self.specs_in_seconds, *other.specs_in_seconds),
        )
        ts.shifts = (*self.shifts, *other.shifts)
        ts.shifts_in_seconds = (*self.shifts_in_seconds, *other.shifts_in_seconds)
        # The following guards against setting units on the result of the addition.
        # Otherwise one could specify time steps in "steps" unit, add that to
        # another TimeStepSpec and reset the units.
        ts.unit_system = "mixed"
        return ts

    def __radd__(self, other):
        return self + other

    def _with_shift(self, shift):
        ts = TimeStepSpec(self)
        if ts.unit_system is None:
            # An unqualified spec adopts the unit of the first shift applied to
            # it, so `TS[::1.e-5] + 2.e-6*TS.seconds` is a seconds-unit spec.
            if shift.unit is TimeStepUnits.SECONDS:
                ts.specs_in_seconds = (*ts.specs_in_seconds, *ts.specs)
                ts.specs = tuple()
                ts.shifts_in_seconds = (*ts.shifts_in_seconds, *ts.shifts)
                ts.shifts = tuple()
            ts.unit_system = shift.unit.value
        # Apply the shift to every part of the spec it was added to.
        ts.shifts = tuple((*pending, shift) for pending in ts.shifts)
        ts.shifts_in_seconds = tuple((*pending, shift) for pending in ts.shifts_in_seconds)
        return ts

    def _accumulate_shifts(self, shifts):
        """
        Split a tuple of pending shifts into a step offset and a seconds offset.

        Keeping the two units apart avoids a lossy seconds round-trip: a shift
        ``N * TimeStepSpec.steps`` on a seconds spec is applied directly to the
        resulting integer steps instead of being converted to ``N * dt`` seconds
        and back.
        """
        steps_offset = 0
        seconds_offset = 0.0
        for shift in shifts:
            if shift.unit is TimeStepUnits.STEPS:
                steps_offset += shift.amount
            else:
                seconds_offset += shift.amount
        return steps_offset, seconds_offset

    def _seconds_to_steps(self, seconds, time_step_size):
        # Convert a shift in seconds into steps, flooring like the lower bound in
        # `_transform_to_steps` so a shift does not silently move an index.
        if seconds == 0:
            return 0
        if time_step_size <= 0:
            raise ValueError(f"Time step size must be strictly positive. You gave {time_step_size}.")
        return floor(seconds / time_step_size)

    @staticmethod
    def _shift_slice(spec, offset):
        # Open ends: a `None` start means "from the beginning", so it moves with
        # the shift; a `None` stop means "to the end" and must stay open.
        if offset == 0:
            return spec
        return slice(
            spec.start + offset if spec.start is not None else offset,
            spec.stop + offset if spec.stop is not None else None,
            spec.step,
        )

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

        Pending shifts (added via the `TimeStepSpec.steps` / `TimeStepSpec.seconds` accessors)
        are resolved here as well, because this is the point where `time_step_size` -- and
        hence the conversion between the two units -- is known. A shift in steps always moves
        the resulting integer indices by exactly that many steps; a shift in seconds is
        converted with the same rounding as the interval bounds.
        """
        if time_step_size <= 0:
            raise ValueError(f"Time step size must be strictly positive. You gave {time_step_size}.")
        specs = []
        for spec, shifts in zip(self.specs, self.shifts):
            steps_offset, seconds_offset = self._accumulate_shifts(shifts)
            offset = steps_offset + self._seconds_to_steps(seconds_offset, time_step_size)
            shifted = self._shift_slice(spec, offset)
            specs.append(self._interpret_negatives(self._interpret_nones(shifted), num_steps))
        for spec, shifts in zip(self.specs_in_seconds, self.shifts_in_seconds):
            steps_offset, seconds_offset = self._accumulate_shifts(shifts)
            shifted = self._shift_slice(spec, seconds_offset)
            for converted in self._transform_to_steps((shifted,), time_step_size):
                # apply a shift in steps directly to the integer index, so that it
                # moves by exactly `steps_offset` steps regardless of `dt`
                converted = self._shift_slice(converted, steps_offset)
                specs.append(self._interpret_negatives(self._interpret_nones(converted), num_steps))
        return PyPIConGPUTimeStepSpec(specs)


# Shorthand for the class above, for the common case that a diagnostic period
# is written inline. `TS[::10]` reads much better than `TimeStepSpec[::10]`.
TS = TimeStepSpec
