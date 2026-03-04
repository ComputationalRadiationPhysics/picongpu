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

from picongpu.picmi.diagnostics import TimeStepSpec

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
                assert list(
                    filter(
                        lambda s: s.start < 0
                        # -1 is allowed as a value for stop only
                        or (s is not None and s.stop < -1)
                        and s.step < 1,
                        ts.get_as_pypicongpu(TIME_STEP_SIZE, INDEX_MAX).specs,
                    )
                ) == []

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
