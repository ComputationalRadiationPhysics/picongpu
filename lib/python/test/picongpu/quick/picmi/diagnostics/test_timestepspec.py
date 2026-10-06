"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase
from functools import reduce
from math import floor, ceil

import pytest

from picongpu.picmi.diagnostics import TS, TimeStepSpec, TimeStepUnits
from picongpu.picmi.diagnostics.timestepspec import TimeStepShift
from picongpu.pypicongpu.output.timestepspec import Spec

# choose larger than any of the numbers used in the TEST_CASES
INDEX_MAX = 200


def inclusive_range(*args):
    """
    Implements range with inclusive endpoint, i.e., in the interval [,] instead of [,).
    """
    args = list(args)
    args[0 if len(args) == 1 else 1] += 1
    return range(*args)


def make_inclusive(spec: slice):
    return slice(spec.start, spec.stop + 1 if spec.stop != -1 else None, spec.step)


def _indices(ts):
    # This function might need to change if the implementation details of
    # TimeStepSpec ever change.
    # It also relies on the picmi object and the pypicongpu object using
    # the same internal variable and storage layout.
    return reduce(
        set.union,
        (list(inclusive_range(INDEX_MAX))[make_inclusive(spec)] for spec in ts.specs),
        set(),
    )


TESTCASES_IN_STEPS = [
    (TimeStepSpec(), set()),
    (TimeStepSpec[:], set(inclusive_range(INDEX_MAX))),
    (TimeStepSpec[::], set(inclusive_range(INDEX_MAX))),
    (TimeStepSpec[10:], set(inclusive_range(10, INDEX_MAX))),
    (TimeStepSpec[10::], set(inclusive_range(10, INDEX_MAX))),
    (TimeStepSpec[:10:], set(inclusive_range(0, 10))),
    (TimeStepSpec[::10], set(inclusive_range(0, INDEX_MAX, 10))),
    (TimeStepSpec[10:20], set(inclusive_range(10, 20))),
    (TimeStepSpec[10:20:], set(inclusive_range(10, 20))),
    (TimeStepSpec[:20:10], set(inclusive_range(0, 20, 10))),
    (TimeStepSpec[20::10], set(inclusive_range(20, INDEX_MAX, 10))),
    (TimeStepSpec[20:50:10], set(inclusive_range(20, 50, 10))),
    (
        TimeStepSpec[20:50:10, ::7],
        set(inclusive_range(20, 50, 10)) | set(inclusive_range(0, INDEX_MAX, 7)),
    ),
    (TimeStepSpec[11], set([11])),
    (TimeStepSpec[11:12, 11], set([11, 12])),
    (TimeStepSpec[10:12, 11], set([10, 11, 12])),
    (
        TimeStepSpec[20:50:10, ::7, 11],
        set(inclusive_range(20, 50, 10)) | set(inclusive_range(0, INDEX_MAX, 7)) | set([11]),
    ),
    (TimeStepSpec[-10:], set(inclusive_range(INDEX_MAX - 10, INDEX_MAX))),
    (TimeStepSpec[:-10:], set(inclusive_range(0, INDEX_MAX - 10))),
    (TimeStepSpec[-10:20], set(inclusive_range(INDEX_MAX - 10, 20))),
    (TimeStepSpec[-10:195], set(inclusive_range(INDEX_MAX - 10, 195))),
    (TimeStepSpec[10:-20], set(inclusive_range(10, INDEX_MAX - 20))),
    (TimeStepSpec[:-20:10], set(inclusive_range(0, INDEX_MAX - 20, 10))),
    (TimeStepSpec[-20::10], set(inclusive_range(INDEX_MAX - 20, INDEX_MAX, 10))),
    (TimeStepSpec[-20:50:10], set(inclusive_range(INDEX_MAX - 20, 50, 10))),
    (TimeStepSpec[-20:190:10], set(inclusive_range(INDEX_MAX - 20, 190, 10))),
    (TimeStepSpec[20:-50:10], set(inclusive_range(20, INDEX_MAX - 50, 10))),
    (
        TimeStepSpec[-20:-50:10],
        set(inclusive_range(INDEX_MAX - 20, INDEX_MAX - 50, 10)),
    ),
    (TimeStepSpec[-11], set([INDEX_MAX - 11])),
]

TESTCASES_IN_STEPS_RAISING = [
    (TimeStepSpec[::-10], set(inclusive_range(0, INDEX_MAX, 10))),
    (TimeStepSpec[:20:-10], set(inclusive_range(0, 20, 10))),
    (TimeStepSpec[20::-10], set(inclusive_range(20, INDEX_MAX, 10))),
    (TimeStepSpec[20:50:-10], set(inclusive_range(20, 50, 10))),
    (TimeStepSpec[-20:50:-10], set(inclusive_range(20, 50, 10))),
    (TimeStepSpec[20:-50:-10], set(inclusive_range(20, 50, 10))),
    (TimeStepSpec[-20:-50:-10], set(inclusive_range(20, 50, 10))),
]

# in seconds (i.e. SI units):
TIME_STEP_SIZE = 0.5
# The following hinge on TIME_STEP_SIZE = 0.5
TESTCASES_IN_SECONDS = [
    (TimeStepSpec()("seconds"), set()),
    (TimeStepSpec[:]("seconds"), set(inclusive_range(INDEX_MAX))),
    (TimeStepSpec[::]("seconds"), set(inclusive_range(INDEX_MAX))),
    (TimeStepSpec[10:]("seconds"), set(inclusive_range(20, INDEX_MAX))),
    (TimeStepSpec[10::]("seconds"), set(inclusive_range(20, INDEX_MAX))),
    (TimeStepSpec[:10:]("seconds"), set(inclusive_range(0, 20))),
    (TimeStepSpec[::10]("seconds"), set(inclusive_range(0, INDEX_MAX, 20))),
    (TimeStepSpec[10:20]("seconds"), set(inclusive_range(20, 40))),
    (TimeStepSpec[10:20:]("seconds"), set(inclusive_range(20, 40))),
    (TimeStepSpec[:20:10]("seconds"), set(inclusive_range(0, 40, 20))),
    (TimeStepSpec[20::10]("seconds"), set(inclusive_range(40, INDEX_MAX, 20))),
    (TimeStepSpec[20:50:10]("seconds"), set(inclusive_range(40, 100, 20))),
    (
        TimeStepSpec[20:50:10, ::7]("seconds"),
        set(inclusive_range(40, 100, 20)) | set(inclusive_range(0, INDEX_MAX, 14)),
    ),
    (TimeStepSpec[11]("seconds"), set([22])),
    (TimeStepSpec[11:12, 11]("seconds"), set([22, 23, 24])),
    (TimeStepSpec[10:12, 11]("seconds"), set(inclusive_range(20, 24))),
    (
        TimeStepSpec[20:50:10, ::7, 11]("seconds"),
        set(inclusive_range(40, 100, 20)) | set(inclusive_range(0, INDEX_MAX, 14)) | set([22]),
    ),
    (TimeStepSpec[-10:]("seconds"), set(inclusive_range(INDEX_MAX - 20, INDEX_MAX))),
    (TimeStepSpec[:-10:]("seconds"), set(inclusive_range(0, INDEX_MAX - 20))),
    (TimeStepSpec[-10:20]("seconds"), set(inclusive_range(INDEX_MAX - 20, 40))),
    (TimeStepSpec[-10:90]("seconds"), set(inclusive_range(INDEX_MAX - 20, 180))),
    (TimeStepSpec[10:-20]("seconds"), set(inclusive_range(20, INDEX_MAX - 40))),
    (TimeStepSpec[:-20:10]("seconds"), set(inclusive_range(0, INDEX_MAX - 40, 20))),
    (
        TimeStepSpec[-20::10]("seconds"),
        set(inclusive_range(INDEX_MAX - 40, INDEX_MAX, 20)),
    ),
    (TimeStepSpec[-20:50:10]("seconds"), set(inclusive_range(INDEX_MAX - 40, 100, 20))),
    (
        TimeStepSpec[-20:90:10]("seconds"),
        set(inclusive_range(INDEX_MAX - 40, 180, 20)),
    ),
    (TimeStepSpec[20:-50:10]("seconds"), set(inclusive_range(40, INDEX_MAX - 100, 20))),
    (
        TimeStepSpec[-20:-50:10]("seconds"),
        set(inclusive_range(INDEX_MAX - 40, INDEX_MAX - 100, 20)),
    ),
    (TimeStepSpec[-11]("seconds"), set([INDEX_MAX - 22])),
]


class TestTimeStepSpec(TestCase):
    def test_get_as_pypicongpu(self):
        """
        The unit conversion is done in get_as_pypicongpu, so we can only test in seconds here.
        """
        for ts, indices in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
            with self.subTest(ts=ts, indices=indices):
                assert _indices(ts.get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX)) == indices

    def test_construct_from_instance(self):
        """
        This tests another branch of the constructor, i.e., a copy constructor.
        """
        for ts, indices in TESTCASES_IN_STEPS:
            with self.subTest(ts=ts, indices=indices):
                assert _indices(TimeStepSpec(ts).get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX)) == indices

    def test_addition_operator(self):
        """
        The unit conversion is done in get_as_pypicongpu, so we can only test in seconds here.
        """
        for ts_steps, indices_steps in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
            for ts_seconds, indices_seconds in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
                ts = ts_steps + ts_seconds
                indices = indices_steps | indices_seconds
                with self.subTest(ts=ts, indices=indices):
                    assert _indices(ts.get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX)) == indices

    def test_dont_reset_unit_from_steps_to_seconds(self):
        ts = TimeStepSpec[:]("steps")
        with pytest.raises(ValueError, match="Don't reset units on a TimeStepSpec. "):
            ts("seconds")

    def test_dont_reset_unit_from_seconds_to_steps(self):
        ts = TimeStepSpec[:]("seconds")
        with pytest.raises(ValueError, match="Don't reset units on a TimeStepSpec. "):
            ts("steps")

    def test_dont_reset_unit_on_addition_result(self):
        with pytest.raises(ValueError, match="Don't reset units on a TimeStepSpec. "):
            (TimeStepSpec[:] + TimeStepSpec[:])("seconds")

    def test_resetting_to_same_unit_is_fine(self):
        with self.subTest(msg="seconds"):
            ts = TimeStepSpec[:]("seconds")
            # not raising an exception
            ts("seconds")

        with self.subTest(msg="steps"):
            ts = TimeStepSpec[:]("steps")
            # not raising an exception
            ts("steps")

    def test_wrong_unit(self):
        with pytest.raises(ValueError, match="Unknown unit in TimeStepSpec."):
            TimeStepSpec[:]("meters")

    def test_raises_on_negative_time_step_size(self):
        with pytest.raises(ValueError, match="Time step size must be strictly positive."):
            TimeStepSpec[:]("seconds").get_as_pypicongpu(-1.0, 10)

    def test_rounding_in_unit_conversion(self):
        # Values are chosen to be sufficiently misaligned such that all special cases are triggered.
        time_step_size = 0.3333
        start = 6.8
        stop = 20.1
        step = 0.7
        ts = TimeStepSpec[start:stop:step]("seconds")
        expected = set(
            filter(
                lambda i: (
                    i >= floor(start / time_step_size)
                    and i < ceil(stop / time_step_size)
                    and (i - floor(start / time_step_size)) % floor(step / time_step_size) == 0
                ),
                inclusive_range(INDEX_MAX),
            )
        )
        assert _indices(ts.get_as_pypicongpu(time_step_size, INDEX_MAX)) == expected

    def test_step_size_smaller_one_in_unit_conversion(self):
        ts = TimeStepSpec[::0.5]("seconds")
        assert _indices(ts.get_as_pypicongpu(0.7, INDEX_MAX)) == set(inclusive_range(INDEX_MAX))

    def test_modify_after_copy_construction(self):
        ts = TimeStepSpec[::0.5]
        ts2 = TimeStepSpec(ts)
        try:
            ts.specs[0] = slice(1, 2, 3)
        except TypeError:
            # It's fine. This is because tuples are immutable to start with.
            pass
        finally:
            assert ts2.specs == (slice(None, None, 0.5),)

    def test_seconds_are_copied(self):
        ts = TimeStepSpec[::0.5]("seconds")
        ts2 = TimeStepSpec(ts)
        assert ts2.specs == ts.specs
        assert ts2.specs_in_seconds == ts.specs_in_seconds

    def test_translation_does_not_contain_negative_numbers(self):
        for ts, indices in TESTCASES_IN_STEPS:
            with self.subTest(ts=ts, indices=indices):
                assert (
                    list(
                        filter(
                            lambda s: s.start < 0
                            # -1 is allowed as a value for stop only
                            or (s is not None and s.stop < -1)
                            and s.step < 1,
                            ts.get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX).specs,
                        )
                    )
                    == []
                )

    def test_raises_for_negative_step_size(self):
        for ts, indices in TESTCASES_IN_STEPS_RAISING:
            with self.subTest(ts=ts, indices=indices):
                with pytest.raises(ValueError, match="Step size must be >= 1"):
                    ts.get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX)

    def test_regression_wrong_int_casting(self):
        stop_time = 1.1195773740290312e-12
        dt = 1.749246958411663e-17
        num_steps = 64004

        as_single = TimeStepSpec[stop_time]("seconds").get_as_pypicongpu(dt, num_steps).specs[0]
        as_slice = TimeStepSpec[stop_time:stop_time]("seconds").get_as_pypicongpu(dt, num_steps).specs[0]
        assert as_single == as_slice


class TestTimeStepShift(TestCase):
    """
    Tests for the unit-shift accessors ``TimeStepSpec.steps`` / ``TimeStepSpec.seconds``.

    The unit conversion and the shift resolution both happen in ``get_as_pypicongpu``.
    With ``time_step_size=1.0`` a step and a second coincide, so the shift-by-seconds
    cases can be checked against the same expected index sets as the shift-by-steps ones.
    """

    DT = 1.0

    def _indices(self, ts, dt=DT):
        return _indices(ts.get_as_pypicongpu(dt, INDEX_MAX))

    def test_shift_by_steps(self):
        ts = TimeStepSpec[::10]("steps") + 5 * TimeStepSpec.steps
        assert self._indices(ts) == set(inclusive_range(5, INDEX_MAX, 10))

    def test_shift_by_steps_bounded(self):
        ts = TimeStepSpec[10:20]("steps") + 5 * TimeStepSpec.steps
        assert self._indices(ts) == set(inclusive_range(15, 25))

    def test_shift_by_seconds(self):
        ts = TimeStepSpec[::10]("seconds") + 5 * TimeStepSpec.seconds
        assert self._indices(ts) == set(inclusive_range(5, INDEX_MAX, 10))

    def test_shift_does_not_move_step(self):
        ts = TimeStepSpec[::10]("steps") + 5 * TimeStepSpec.steps
        assert ts.get_as_pypicongpu(self.DT, INDEX_MAX).specs == [
            # the step (10) is unchanged, only start moved
            Spec(start=5, stop=-1, step=10)
        ]

    def test_open_start_is_shifted(self):
        ts = TimeStepSpec[::7]("steps") + 3 * TimeStepSpec.steps
        assert self._indices(ts) == set(inclusive_range(3, INDEX_MAX, 7))

    def test_open_stop_stays_open(self):
        ts = TimeStepSpec[2:]("steps") + 4 * TimeStepSpec.steps
        assert self._indices(ts) == set(inclusive_range(6, INDEX_MAX))
        # an explicit stop moves with the shift
        ts2 = TimeStepSpec[2:9]("steps") + 4 * TimeStepSpec.steps
        assert self._indices(ts2) == set(inclusive_range(6, 13))

    def test_shift_is_deferred_to_get_as_pypicongpu(self):
        # The seconds shift is unresolved until translation, where dt is known.
        ts = TimeStepSpec[::1e-5] + 2.0e-6 * TimeStepSpec.seconds
        assert self._indices(ts, dt=1.0e-6) == set(inclusive_range(2, INDEX_MAX, 10))
        # With half the time step the same physical shift covers twice the steps.
        assert self._indices(ts, dt=0.5e-6) == set(inclusive_range(4, INDEX_MAX, 20))

    def test_shift_steps_applied_to_seconds_spec(self):
        # a shift measured in steps on a seconds spec is dt-dependent
        ts = TimeStepSpec[::1e-5]("seconds") + 2 * TimeStepSpec.steps
        assert self._indices(ts, dt=1.0e-6) == set(inclusive_range(2, INDEX_MAX, 10))

    def test_steps_shift_on_seconds_spec_does_not_drift(self):
        # `3*dt/dt` round-trips to 2.999...: a step shift must not silently move
        # the index; it moves by exactly three steps.
        ts = TimeStepSpec[::1e-1]("seconds") + 3 * TimeStepSpec.steps
        assert ts.get_as_pypicongpu(0.1, INDEX_MAX).specs == [Spec(start=3, stop=-1, step=1)]

    def test_seconds_shift_rounds_to_nearest(self):
        # 0.3 s at dt = 0.1 s is nominally 3 steps, but 0.3/0.1 == 2.999...
        # in binary floating point: the shift must round to three, agreeing
        # with a plain interval bound, not floor to two.
        ts = TimeStepSpec[::1e-1]("seconds") + 0.3 * TimeStepSpec.seconds
        assert ts.get_as_pypicongpu(0.1, INDEX_MAX).specs == [Spec(start=3, stop=-1, step=1)]

    def test_seconds_shift_rounding_is_sign_symmetric(self):
        # +0.3 s and -0.3 s must move by +3 and -3 steps, not +2/-2 (int) or
        # +2/-3 (floor). The two paths (steps spec and seconds spec) agree.
        for spec_unit in ("steps", "seconds"):
            with self.subTest(spec_unit=spec_unit):
                pos = TimeStepSpec[::1](spec_unit) + 0.3 * TimeStepSpec.seconds
                neg = TimeStepSpec[::1](spec_unit) + (-0.3) * TimeStepSpec.seconds
                # INDEX_MAX == 200: an open-start spec shifted down by 3 starts at 197.
                assert pos.get_as_pypicongpu(0.1, INDEX_MAX).specs[0].start == 3
                assert neg.get_as_pypicongpu(0.1, INDEX_MAX).specs[0].start == 197

    def test_seconds_shift_matches_across_spec_units(self):
        # The two resolution paths (a steps spec and a seconds spec) must agree
        # on the step offset for both signs.
        for shift in (0.3, -0.3):
            with self.subTest(shift=shift):
                steps_spec = (TimeStepSpec[10:10]("steps") + shift * TimeStepSpec.seconds).get_as_pypicongpu(
                    0.1, INDEX_MAX
                )
                seconds_spec = (TimeStepSpec[1.0:1.0]("seconds") + shift * TimeStepSpec.seconds).get_as_pypicongpu(
                    0.1, INDEX_MAX
                )
                # `[10:10]` steps == `[1.0:1.0]` seconds at dt = 0.1
                assert steps_spec.specs[0].start == seconds_spec.specs[0].start

    def test_unqualified_spec_adopts_shift_unit(self):
        seconds_rest = TimeStepSpec[::1e-5] + 2.0e-6 * TimeStepSpec.seconds
        assert seconds_rest.unit_system == "seconds"
        assert seconds_rest.specs == tuple()
        assert seconds_rest.specs_in_seconds == (slice(None, None, 1e-5),)

        steps_rest = TimeStepSpec[::10] + 5 * TimeStepSpec.steps
        assert steps_rest.unit_system == "steps"
        assert steps_rest.specs_in_seconds == tuple()
        assert steps_rest.specs == (slice(None, None, 10),)

    def test_shifts_are_unit_tagged_not_merged(self):
        # steps and seconds shifts on different parts of a mixed spec stay separate
        ts = (TimeStepSpec[::10]("steps") + 5 * TimeStepSpec.steps) + (
            TimeStepSpec[::1e-5] + 2.0e-6 * TimeStepSpec.seconds
        )
        assert ts.unit_system == "mixed"
        assert ts.specs == (slice(None, None, 10),)
        assert ts.specs_in_seconds == (slice(None, None, 1e-5),)
        assert ts.shifts == ((TimeStepShift(5, "steps"),),)
        assert ts.shifts_in_seconds == ((TimeStepShift(2.0e-6, "seconds"),),)

    def test_shift_does_not_mutate_original(self):
        ts = TimeStepSpec[::10]("steps")
        shifted = ts + 5 * TimeStepSpec.steps
        assert ts.shifts == (tuple(),)
        assert self._indices(ts) == set(inclusive_range(0, INDEX_MAX, 10))
        assert self._indices(shifted) == set(inclusive_range(5, INDEX_MAX, 10))

    def test_shift_does_not_reset_units(self):
        ts = TimeStepSpec[::10]("steps") + 5 * TimeStepSpec.steps
        with pytest.raises(ValueError, match="Don't reset units on a TimeStepSpec."):
            ts("seconds")

    def test_documented_interface_expression(self):
        ts = (TS[::1]("steps") + 10 * TS.steps) + (TS[::1.0e-5] + 2.0e-6 * TS.seconds)
        assert ts.unit_system == "mixed"
        assert self._indices(ts, dt=1.0e-6) == set(inclusive_range(10, INDEX_MAX)) | set(
            inclusive_range(2, INDEX_MAX, 10)
        )

    def test_shift_by_zero_is_identity(self):
        # NB: the expected sets in TESTCASES_IN_SECONDS assume TIME_STEP_SIZE.
        for ts, indices in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
            with self.subTest(ts=ts):
                shifted = ts + 0 * TimeStepSpec.steps
                assert _indices(shifted.get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX)) == indices

    def test_unknown_unit_shift(self):
        with pytest.raises(ValueError, match="Unknown time step unit."):
            TimeStepShift(1, "meters")

    def test_units_membership_is_case_insensitive(self):
        # Public `TimeStepUnits` is exported; `in` must stay case-insensitive
        # (the metaclass shim provided this before the StrEnum swap).
        for value in ("steps", "STEPS", "Steps", "seconds", "SECONDS", "Seconds"):
            with self.subTest(value=value):
                assert value in TimeStepUnits
        assert "meters" not in TimeStepUnits
        assert TimeStepUnits.STEPS in TimeStepUnits

    def test_dont_reset_shifted_seconds_unit(self):
        ts = TimeStepSpec[::10] + 5 * TimeStepSpec.seconds
        with pytest.raises(ValueError, match="Don't reset units on a TimeStepSpec."):
            ts("steps")
