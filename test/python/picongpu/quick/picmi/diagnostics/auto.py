"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import Auto, TimeStepSpec
from picongpu.pypicongpu.output.auto import Auto as PyPIConGPUAuto
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec
import unittest


# Test cases for valid Auto inputs
TESTCASES_VALID = [
    (10, [{"start": 0, "stop": 199, "step": 10}]),
    (0, []),
    (TimeStepSpec([slice(None, None, 10)]), [{"start": 0, "stop": 199, "step": 10}]),
    (TimeStepSpec([5, 10]), [{"start": 5, "stop": 6, "step": 1}, {"start": 10, "stop": 11, "step": 1}]),
    (TimeStepSpec([slice(-10, None, 1)]), [{"start": 190, "stop": 199, "step": 1}]),
    (TimeStepSpec(), []),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    ("invalid", "period must be an integer or TimeStepSpec"),
    (-10, "period must be non-negative"),
]

# Invalid test cases for TimeStepSpec with negative steps
TESTCASES_INVALID_TIMESTEPS = [
    (TimeStepSpec([slice(None, None, -10)]), "Step size must be >= 1"),
    (TimeStepSpec([slice(5, 10, -2)]), "Step size must be >= 1"),
]

# Test cases for warning when period is disabled
TESTCASES_WARNING = [
    (0, "Auto output is disabled because period is set to 0 or an empty TimeStepSpec"),
    (TimeStepSpec(), "Auto output is disabled because period is set to 0 or an empty TimeStepSpec"),
]


class PICMI_TestAuto(unittest.TestCase):
    def test_auto_instantiation(self):
        """Test Auto instantiation and validation."""
        for period, _ in TESTCASES_VALID:
            with self.subTest(period=period):
                auto = Auto(period=period)
                if isinstance(period, int):
                    expected = TimeStepSpec[::period]("steps") if period > 0 else TimeStepSpec()("steps")
                    self.assertEqual(
                        auto.period.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                        expected.get_as_pypicongpu(0.5, 200).get_rendering_context(),
                    )
                else:
                    self.assertEqual(auto.period, period)
                if not period or (
                    isinstance(period, TimeStepSpec)
                    and not period.get_as_pypicongpu(0.5, 200).get_rendering_context().get("specs", [])
                ):
                    with self.assertWarnsRegex(UserWarning, "Auto output is disabled"):
                        auto.check()
                else:
                    auto.check()  # Should not raise or warn

        for period, expected_error in TESTCASES_INVALID:
            with self.subTest(period=period, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, TypeError), expected_error):
                    Auto(period=period).check()

    def test_auto_serialization(self):
        """Test Auto serialization to PyPIConGPUAuto."""
        for period, expected_specs in TESTCASES_VALID:
            with self.subTest(period=period, expected_specs=expected_specs):
                auto = Auto(period=period)
                pypicongpu_auto = auto.get_as_pypicongpu({}, 0.5, 200)
                self.assertIsInstance(pypicongpu_auto, PyPIConGPUAuto)
                self.assertIsInstance(pypicongpu_auto.period, PyPIConGPUTimeStepSpec)
                serialized = pypicongpu_auto.get_rendering_context()
                self.assertTrue(serialized["typeID"]["auto"])
                self.assertEqual(serialized["data"]["period"]["specs"], expected_specs)
                self.assertEqual(serialized["data"]["png_axis"], [{"axis": "yx"}, {"axis": "yz"}])

    def test_auto_warning(self):
        """Test warning for disabled Auto output."""
        for period, expected_warning in TESTCASES_WARNING:
            with self.subTest(period=period, expected_warning=expected_warning):
                auto = Auto(period=period)
                with self.assertWarnsRegex(UserWarning, expected_warning):
                    auto.check()

    def test_auto_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for period, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(period=period, expected_error=expected_error):
                auto = Auto(period=period)
                with self.assertRaisesRegex(ValueError, expected_error):
                    auto.get_as_pypicongpu({}, 0.5, 200)

    def test_auto_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        auto = Auto(period=10)
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            auto.get_as_pypicongpu({}, -0.5, 200)
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            auto.get_as_pypicongpu({}, 0, 200)


if __name__ == "__main__":
    unittest.main()
