"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

import unittest
from math import floor, ceil

from picongpu.picmi.diagnostics import TimeStepSpec

INDEX_MAX = 200


def inclusive_range(*args):
    """
    Implements range with inclusive endpoint, i.e., in the interval [,] instead of [,).
    """
    args = list(args)
    args[0 if len(args) == 1 else 1] += 1
    return range(*args)


def _indices(ts, num_steps=INDEX_MAX):
    pypicongpu_spec = ts.get_as_pypicongpu(0.5, num_steps)
    steps = set()
    for spec in pypicongpu_spec.specs:
        start = spec.start or 0
        stop = spec.stop + 1 if spec.stop is not None else num_steps
        step = spec.step or 1
        steps.update(range(start, stop, step))
    return steps


TESTCASES_IN_STEPS = [
    (TimeStepSpec(), set()),
    (TimeStepSpec([slice(None, None, 1)]), set(range(INDEX_MAX))),
    (TimeStepSpec([slice(None, None, 1)]), set(range(INDEX_MAX))),
    (TimeStepSpec([slice(10, None, 1)]), set(range(10, INDEX_MAX))),
    (TimeStepSpec([slice(10, None, 1)]), set(range(10, INDEX_MAX))),
    (TimeStepSpec([slice(0, 10, 1)]), set(range(0, 10))),
    (TimeStepSpec([slice(None, None, 10)]), set(range(0, INDEX_MAX, 10))),
    (TimeStepSpec([slice(10, 20, 1)]), set(range(10, 20))),
    (TimeStepSpec([slice(10, 20, 1)]), set(range(10, 20))),
    (TimeStepSpec([slice(0, 20, 10)]), set(range(0, 20, 10))),
    (TimeStepSpec([slice(20, None, 10)]), set(range(20, INDEX_MAX, 10))),
    (TimeStepSpec([slice(20, 50, 10)]), set(range(20, 50, 10))),
    (
        TimeStepSpec([slice(20, 50, 10), slice(None, None, 7)]),
        set(range(20, 50, 10)) | set(range(0, INDEX_MAX, 7)),
    ),
    (TimeStepSpec([11]), set([11])),
    (TimeStepSpec([slice(11, 12, 1), 11]), set([11, 12])),
    (TimeStepSpec([slice(10, 12, 1), 11]), set([10, 11, 12])),
    (
        TimeStepSpec([slice(20, 50, 10), slice(None, None, 7), 11]),
        set(range(20, 50, 10)) | set(range(0, INDEX_MAX, 7)) | set([11]),
    ),
    (TimeStepSpec([slice(-10, None, 1)]), set(range(INDEX_MAX - 10, INDEX_MAX))),
    (TimeStepSpec([slice(0, -10, 1)]), set(range(0, INDEX_MAX - 10))),
    (TimeStepSpec([slice(-10, 20, 1)]), set(range(INDEX_MAX - 10, 20))),
    (TimeStepSpec([slice(-10, 195, 1)]), set(range(INDEX_MAX - 10, 195))),
    (TimeStepSpec([slice(10, -20, 1)]), set(range(10, INDEX_MAX - 20))),
    (TimeStepSpec([slice(0, -20, 10)]), set(range(0, INDEX_MAX - 20, 10))),
    (TimeStepSpec([slice(-20, None, 10)]), set(range(INDEX_MAX - 20, INDEX_MAX, 10))),
    (TimeStepSpec([slice(-20, 50, 10)]), set(range(INDEX_MAX - 20, 50, 10))),
    (TimeStepSpec([slice(-20, 190, 10)]), set(range(INDEX_MAX - 20, 190, 10))),
    (TimeStepSpec([slice(20, -50, 10)]), set(range(20, INDEX_MAX - 50, 10))),
    (
        TimeStepSpec([slice(-20, -50, 10)]),
        set(range(INDEX_MAX - 20, INDEX_MAX - 50, 10)),
    ),
    (TimeStepSpec([-11]), set([INDEX_MAX - 11])),
]

TESTCASES_IN_SECONDS = [
    (TimeStepSpec(unit="seconds"), set()),
    (TimeStepSpec([slice(None, None, 1)], unit="seconds"), set(range(INDEX_MAX))),
    (TimeStepSpec([slice(None, None, 1)], unit="seconds"), set(range(INDEX_MAX))),
    (TimeStepSpec([slice(10, None, 1)], unit="seconds"), set(range(20, INDEX_MAX))),
    (TimeStepSpec([slice(10, None, 1)], unit="seconds"), set(range(20, INDEX_MAX))),
    (TimeStepSpec([slice(0, 10, 1)], unit="seconds"), set(range(0, 20))),
    (TimeStepSpec([slice(None, None, 10)], unit="seconds"), set(range(0, INDEX_MAX, 20))),
    (TimeStepSpec([slice(10, 20, 1)], unit="seconds"), set(range(20, 40))),
    (TimeStepSpec([slice(10, 20, 1)], unit="seconds"), set(range(20, 40))),
    (TimeStepSpec([slice(0, 20, 10)], unit="seconds"), set(range(0, 40, 20))),
    (TimeStepSpec([slice(20, None, 10)], unit="seconds"), set(range(40, INDEX_MAX, 20))),
    (TimeStepSpec([slice(20, 50, 10)], unit="seconds"), set(range(40, 100, 20))),
    (
        TimeStepSpec([slice(20, 50, 10), slice(None, None, 7)], unit="seconds"),
        set(range(40, 100, 20)) | set(range(0, INDEX_MAX, 14)),
    ),
    (TimeStepSpec([11], unit="seconds"), set([22])),
    (TimeStepSpec([slice(11, 12, 1), 11], unit="seconds"), set([22, 23, 24])),
    (TimeStepSpec([slice(10, 12, 1), 11], unit="seconds"), set(range(20, 24))),
    (
        TimeStepSpec([slice(20, 50, 10), slice(None, None, 7), 11], unit="seconds"),
        set(range(40, 100, 20)) | set(range(0, INDEX_MAX, 14)) | set([22]),
    ),
    (TimeStepSpec([slice(-10, None, 1)], unit="seconds"), set(range(INDEX_MAX - 20, INDEX_MAX))),
    (TimeStepSpec([slice(0, -10, 1)], unit="seconds"), set(range(0, INDEX_MAX - 20))),
    (TimeStepSpec([slice(-10, 20, 1)], unit="seconds"), set(range(INDEX_MAX - 20, 40))),
    (TimeStepSpec([slice(-10, 90, 1)], unit="seconds"), set(range(INDEX_MAX - 20, 180))),
    (TimeStepSpec([slice(10, -20, 1)], unit="seconds"), set(range(20, INDEX_MAX - 40))),
    (TimeStepSpec([slice(0, -20, 10)], unit="seconds"), set(range(0, INDEX_MAX - 40, 20))),
    (
        TimeStepSpec([slice(-20, None, 10)], unit="seconds"),
        set(range(INDEX_MAX - 40, INDEX_MAX, 20)),
    ),
    (TimeStepSpec([slice(-20, 50, 10)], unit="seconds"), set(range(INDEX_MAX - 40, 100, 20))),
    (
        TimeStepSpec([slice(-20, 90, 10)], unit="seconds"),
        set(range(INDEX_MAX - 40, 180, 20)),
    ),
    (TimeStepSpec([slice(20, -50, 10)], unit="seconds"), set(range(40, INDEX_MAX - 100, 20))),
    (
        TimeStepSpec([slice(-20, -50, 10)], unit="seconds"),
        set(range(INDEX_MAX - 40, INDEX_MAX - 100, 20)),
    ),
    (TimeStepSpec([-11], unit="seconds"), set([INDEX_MAX - 22])),
]

TESTCASES_IN_STEPS_RAISING = [
    (TimeStepSpec([slice(None, None, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(0, 20, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(20, None, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(20, 50, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(-20, 50, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(20, -50, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(-20, -50, -10)]), "Step size must be >= 1"),
]


class TestTimeStepSpec(unittest.TestCase):
    def test_get_as_pypicongpu(self):
        """Test conversion to pypicongpu TimeStepSpec."""
        for ts, indices in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
            with self.subTest(ts=ts, indices=indices):
                self.assertEqual(
                    _indices(ts, INDEX_MAX),
                    indices,
                )

    def test_construct_from_instance(self):
        """Test copy constructor."""
        for ts, indices in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
            with self.subTest(ts=ts, indices=indices):
                self.assertEqual(
                    _indices(TimeStepSpec(ts), INDEX_MAX),
                    indices,
                )

    def test_addition_operator(self):
        """Test addition operator for combining TimeStepSpec instances."""
        for ts_steps, indices_steps in TESTCASES_IN_STEPS:
            for ts_seconds, indices_seconds in TESTCASES_IN_SECONDS:
                ts = ts_steps + ts_seconds
                indices = indices_steps | indices_seconds
                with self.subTest(ts=ts, indices=indices):
                    self.assertEqual(
                        _indices(ts, INDEX_MAX),
                        indices,
                    )

    def test_dont_reset_unit_from_steps_to_seconds(self):
        """Test that resetting unit from steps to seconds raises an error."""
        ts = TimeStepSpec([slice(None, None, 1)], unit="steps")
        with self.assertRaisesRegex(ValueError, "Don't reset units on a TimeStepSpec"):
            ts(unit="seconds")

    def test_dont_reset_unit_from_seconds_to_steps(self):
        """Test that resetting unit from seconds to steps raises an error."""
        ts = TimeStepSpec([slice(None, None, 1)], unit="seconds")
        with self.assertRaisesRegex(ValueError, "Don't reset units on a TimeStepSpec"):
            ts(unit="steps")

    def test_dont_reset_unit_on_addition_result(self):
        """Test that resetting unit on addition result raises an error."""
        ts = TimeStepSpec([slice(None, None, 1)]) + TimeStepSpec([slice(None, None, 1)])
        with self.assertRaisesRegex(ValueError, "Don't reset units on a TimeStepSpec"):
            ts(unit="seconds")

    def test_resetting_to_same_unit_is_fine(self):
        """Test that resetting to the same unit is allowed."""
        with self.subTest(msg="seconds"):
            ts = TimeStepSpec([slice(None, None, 1)], unit="seconds")
            ts(unit="seconds")  # Should not raise
        with self.subTest(msg="steps"):
            ts = TimeStepSpec([slice(None, None, 1)], unit="steps")
            ts(unit="steps")  # Should not raise

    def test_wrong_unit(self):
        """Test that an invalid unit raises an error."""
        with self.assertRaisesRegex(ValueError, "Unknown unit in TimeStepSpec"):
            TimeStepSpec([slice(None, None, 1)], unit="meters")

    def test_raises_on_negative_time_step_size(self):
        """Test that negative time step size raises an error."""
        ts = TimeStepSpec([slice(None, None, 1)], unit="seconds")
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            ts.get_as_pypicongpu(-1.0, INDEX_MAX)

    def test_rounding_in_unit_conversion(self):
        """Test rounding behavior in unit conversion."""
        time_step_size = 0.3333
        start = 6.8
        stop = 20.1
        step = 0.7
        ts = TimeStepSpec([slice(start, stop, step)], unit="seconds")
        expected = set(
            range(
                floor(start / time_step_size),
                ceil(stop / time_step_size),
                max(1, floor(step / time_step_size)),
            )
        )
        self.assertEqual(_indices(ts, INDEX_MAX), expected)

    def test_step_size_smaller_one_in_unit_conversion(self):
        """Test handling of step size smaller than one in unit conversion."""
        ts = TimeStepSpec([slice(None, None, 0.5)], unit="seconds")
        self.assertEqual(
            _indices(ts, INDEX_MAX),
            set(range(INDEX_MAX)),
        )

    def test_modify_after_copy_construction(self):
        """Test that modifying specs after copy construction does not affect the copy."""
        ts = TimeStepSpec([slice(None, None, 0.5)])
        ts2 = TimeStepSpec(ts)
        try:
            ts.specs[0] = slice(1, 2, 3)
        except TypeError:
            pass  # Expected due to tuple immutability
        self.assertEqual(ts2.specs, [slice(None, None, 0.5)])

    def test_seconds_are_copied(self):
        """Test that unit is copied in copy constructor."""
        ts = TimeStepSpec([slice(None, None, 0.5)], unit="seconds")
        ts2 = TimeStepSpec(ts)
        self.assertEqual(ts2.specs, ts.specs)
        self.assertEqual(ts2.unit, ts.unit)

    def test_translation_does_not_contain_negative_numbers(self):
        """Test that translated specs do not contain negative numbers."""
        for ts, indices in TESTCASES_IN_STEPS + TESTCASES_IN_SECONDS:
            with self.subTest(ts=ts, indices=indices):
                pypicongpu_spec = ts.get_as_pypicongpu(0.5, INDEX_MAX)
                self.assertEqual(
                    [
                        s
                        for s in pypicongpu_spec.specs
                        if s.start is not None
                        and s.start < 0
                        or (s.stop is not None and s.stop < 0)
                        or (s.step is not None and s.step < 1)
                    ],
                    [],
                )

    def test_raises_for_negative_step_size(self):
        """Test that negative step sizes raise an error."""
        for ts, error_msg in TESTCASES_IN_STEPS_RAISING:
            with self.subTest(ts=ts, error_msg=error_msg):
                with self.assertRaisesRegex(ValueError, error_msg):
                    ts.get_as_pypicongpu(0.5, INDEX_MAX)

    def test_regression_wrong_int_casting(self):
        """Test regression for correct integer casting in unit conversion."""
        stop_time = 1.1195773740290312e-12
        dt = 1.749246958411663e-17
        num_steps = 64004
        as_single = TimeStepSpec([stop_time], unit="seconds").get_as_pypicongpu(dt, num_steps).specs[0]
        as_slice = (
            TimeStepSpec([slice(stop_time, stop_time + dt, 1)], unit="seconds")
            .get_as_pypicongpu(dt, num_steps)
            .specs[0]
        )
        self.assertEqual(as_single, as_slice)


if __name__ == "__main__":
    unittest.main()
