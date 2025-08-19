"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
import unittest
import math


INDEX_MAX = 400
NUM_STEPS = 200


def _indices(ts: TimeStepSpec, num_steps: int) -> set:
    """Helper function to get all indices from a TimeStepSpec."""
    pypicongpu_spec = ts.get_as_pypicongpu(0.5, num_steps)
    indices = set()
    for spec in pypicongpu_spec.specs:
        start = spec.start or 0
        stop = spec.stop or num_steps
        step = spec.step or 1
        indices.update(range(start, stop, step))
    return indices


class TestTimeStepSpec(unittest.TestCase):
    def test_parse(self):
        """Test parsing of slice and index specifications."""
        test_cases = [
            (TimeStepSpec[0:100:2]("steps"), [slice(0, 100, 2)]),
            (TimeStepSpec[100]("steps"), [slice(100, 101, 1)]),
            (TimeStepSpec[0:100:2, 200:300:5]("steps"), [slice(0, 100, 2), slice(200, 300, 5)]),
            (TimeStepSpec[0.5:1.5:0.5]("seconds"), [slice(0.5, 1.5, 0.5)]),
            (TimeStepSpec()("steps"), []),
        ]
        for ts, expected_specs in test_cases:
            with self.subTest(ts=ts):
                self.assertEqual(list(ts.specs + ts.specs_in_seconds), expected_specs)

    def test_parse_invalid(self):
        """Test invalid specifications."""
        test_cases = [
            (TimeStepSpec[0:100:-1], "Step size must be >= 1"),
        ]
        for spec, expected_error in test_cases:
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, expected_error):
                    spec("steps").get_as_pypicongpu(0.5, 200)

    def test_add(self):
        """Test addition of TimeStepSpec objects."""
        ts1 = TimeStepSpec[0:100:2]("steps")
        ts2 = TimeStepSpec[200:300:5]("steps")
        ts3 = ts1 + ts2
        self.assertEqual(ts3.specs, (slice(0, 100, 2), slice(200, 300, 5)))
        self.assertEqual(ts3.unit_system, "steps")
        ts4 = TimeStepSpec[100]("steps")
        ts5 = ts1 + ts4
        self.assertEqual(ts5.specs, (slice(0, 100, 2), slice(100, 101, 1)))

    def test_add_invalid(self):
        """Test addition with different units."""
        ts1 = TimeStepSpec[0:100:2]("steps")
        ts2 = TimeStepSpec[0:100:2]("seconds")
        with self.assertRaisesRegex(ValueError, "Cannot add TimeStepSpec objects with different units"):
            ts1 + ts2

    def test_get_as_pypicongpu(self):
        """Test conversion to pypicongpu TimeStepSpec."""
        test_cases = [
            (TimeStepSpec[160:180:10]("steps"), {160, 170}),
            (TimeStepSpec[160]("steps"), {160}),
            (TimeStepSpec[40:100:20]("steps"), {40, 60, 80}),
            (TimeStepSpec[178:190:12]("steps"), {178}),
            (TimeStepSpec()("steps"), set()),
            (TimeStepSpec[-10:-1:1]("steps"), {190, 191, 192, 193, 194, 195, 196, 197, 198}),
            ((TimeStepSpec[0:100:2])("seconds"), set(range(0, 200, 4))),
            (TimeStepSpec[0.5]("seconds"), {1}),
            (TimeStepSpec[0:400:1]("steps"), set(range(0, 200))),
        ]
        for ts, indices in test_cases:
            with self.subTest(ts=ts, indices=indices):
                self.assertEqual(
                    _indices(ts, NUM_STEPS),
                    indices & set(range(NUM_STEPS)),
                )

    def test_invalid_arguments(self):
        """Test invalid arguments in get_as_pypicongpu."""
        ts = TimeStepSpec[0:100:2]("steps")
        for time_step_size, num_steps, expected_error in [
            (0.0, 200, "time_step_size must be positive"),
            (-1.0, 200, "time_step_size must be positive"),
            (1.0, 0, "num_steps must be positive"),
            (1.0, -1, "num_steps must be positive"),
        ]:
            with self.subTest(time_step_size=time_step_size, num_steps=num_steps):
                with self.assertRaisesRegex(ValueError, expected_error):
                    ts.get_as_pypicongpu(time_step_size, num_steps)

    def test_wrong_unit(self):
        """Test that an invalid unit raises an error."""
        with self.assertRaisesRegex(ValueError, "Unknown unit in TimeStepSpec"):
            TimeStepSpec[0:100:2]("invalid")

    def test_resetting_to_same_unit_is_fine(self):
        """Test that resetting to the same unit is allowed."""
        for unit in ["seconds", "steps"]:
            with self.subTest(unit=unit):
                ts = TimeStepSpec[0:100:2](unit)
                ts(unit)  # Should not raise

    def test_rounding_in_unit_conversion(self):
        """Test rounding behavior in unit conversion."""
        ts = TimeStepSpec[0:5:0.4]("seconds")
        self.assertEqual(_indices(ts, INDEX_MAX), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_step_size_smaller_one_in_unit_conversion(self):
        """Test handling of step size smaller than one in unit conversion."""
        ts = TimeStepSpec[0:5:0.1]("seconds")
        self.assertEqual(_indices(ts, INDEX_MAX), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_seconds_are_copied(self):
        """Test that unit is copied in copy constructor."""
        ts = TimeStepSpec[0:100:2]("seconds")
        ts2 = TimeStepSpec(ts)
        self.assertEqual(ts2.unit_system, "seconds")
        self.assertEqual(ts2.specs_in_seconds, ts.specs_in_seconds)

    def test_regression_wrong_int_casting(self):
        """Test regression for correct integer casting in unit conversion."""
        dt = 0.5
        num_steps = 200
        for stop_time in [0.0, 0.5, 1.0, 1.5, 2.0]:
            with self.subTest(stop_time=stop_time):
                as_single = TimeStepSpec[stop_time]("seconds").get_as_pypicongpu(dt, num_steps).specs[0]
                expected = slice(math.floor(stop_time / dt), math.floor(stop_time / dt) + 1, 1)
                self.assertEqual(as_single, expected)


if __name__ == "__main__":
    unittest.main()
