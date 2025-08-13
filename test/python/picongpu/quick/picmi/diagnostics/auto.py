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
    (TimeStepSpec[::10], [{"start": 0, "stop": -1, "step": 10}]),
    (TimeStepSpec[5, 10], [{"start": 5, "stop": 5, "step": 1}, {"start": 10, "stop": 10, "step": 1}]),
    (TimeStepSpec(), []),  # Empty TimeStepSpec
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (10, "period must be a TimeStepSpec"),
    ("invalid", "period must be a TimeStepSpec"),
]

# Invalid test cases for TimeStepSpec with negative steps
TESTCASES_INVALID_TIMESTEPS = [
    (TimeStepSpec[::-10], "Step size must be >= 1"),
    (TimeStepSpec[5:10:-2], "Step size must be >= 1"),
]


class TestAuto(unittest.TestCase):
    def test_auto_instantiation(self):
        """Test Auto instantiation and validation."""
        for period, _ in TESTCASES_VALID:
            with self.subTest(period=period):
                auto = Auto(period=period)
                self.assertEqual(auto.period, period)
                auto.check()  # Should not raise

        for period, expected_error in TESTCASES_INVALID:
            with self.subTest(period=period, expected_error=expected_error):
                with self.assertRaisesRegex(TypeError, expected_error):
                    Auto(period=period)

    def test_auto_serialization(self):
        """Test Auto serialization to PyPIConGPUAuto."""
        for period, expected_specs in TESTCASES_VALID:
            with self.subTest(period=period, expected_specs=expected_specs):
                auto = Auto(period=period)
                pypicongpu_auto = auto.get_as_pypicongpu({}, 0.5, 100)
                self.assertIsInstance(pypicongpu_auto, PyPIConGPUAuto)
                self.assertIsInstance(pypicongpu_auto.period, PyPIConGPUTimeStepSpec)
                serialized = pypicongpu_auto.get_rendering_context()
                self.assertEqual(serialized["period"]["specs"], expected_specs)

    def test_auto_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for period, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(period=period, expected_error=expected_error):
                auto = Auto(period=period)
                with self.assertRaisesRegex(ValueError, expected_error):
                    auto.get_as_pypicongpu({}, 0.5, 100)

    def test_auto_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        auto = Auto(period=TimeStepSpec[::10])
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            auto.get_as_pypicongpu({}, -0.5, 100)
        with self.assertRaisesRegex(ValueError, "Time step size must be strictly positive"):
            auto.get_as_pypicongpu({}, 0, 100)


if __name__ == "__main__":
    unittest.main()
