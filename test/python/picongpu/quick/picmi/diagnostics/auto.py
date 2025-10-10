"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard
from picongpu.picmi.diagnostics import Auto, TimeStepSpec
from picongpu.pypicongpu.output.auto import Auto as PyPIConGPUAuto
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec


TESTCASES_VALID = [
    (10, [{"start": 0, "stop": -1, "step": 10}]),
    (TimeStepSpec(slice(None, None, 10)), [{"start": 0, "stop": -1, "step": 10}]),
]

TESTCASES_INVALID = [
    ("invalid", "period must be an integer or TimeStepSpec"),
    (-10, "period must be non-negative"),
]


class PICMI_TestAuto(unittest.TestCase):
    def test_valid_periods(self):
        """Test Auto instantiation, validation, and conversion."""
        for period, expected_specs in TESTCASES_VALID:
            with self.subTest(period=period):
                auto = Auto(period=period)
                self.assertIsInstance(auto.period, TimeStepSpec)
                auto.check()

                # Convert to PyPIConGPUAuto
                pypicongpu_auto = auto.get_as_pypicongpu(0.5, 200)
                self.assertIsInstance(pypicongpu_auto, PyPIConGPUAuto)
                self.assertIsInstance(pypicongpu_auto.period, PyPIConGPUTimeStepSpec)

                # Validate rendered specs
                serialized = pypicongpu_auto.get_rendering_context()
                self.assertTrue(serialized["typeID"]["auto"])
                self.assertEqual(serialized["data"]["period"]["specs"], expected_specs)

    def test_invalid_period_type(self):
        """Test invalid input types."""
        with self.assertRaises(typeguard.TypeCheckError):
            Auto(period="invalid")

    def test_negative_period(self):
        """Test negative integer period raises error."""
        with self.assertRaisesRegex(ValueError, "period must be non-negative"):
            Auto(period=-5)

    def test_check_invalid_period(self):
        """Test that check() catches wrong type."""
        auto = Auto(period=10)
        auto.period = "invalid"
        with self.assertRaisesRegex(TypeError, "period must be a TimeStepSpec"):
            auto.check()


if __name__ == "__main__":
    unittest.main()
