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
import typeguard


TESTCASES_VALID = [
    (10, [{"start": 0, "stop": 199, "step": 10}]),
    (TimeStepSpec([slice(None, None, 10)]), [{"start": 0, "stop": 199, "step": 10}]),
]

TESTCASES_INVALID = [
    ("invalid", "period must be an integer or TimeStepSpec"),
    (-10, "period must be non-negative"),
]

TESTCASES_INVALID_TIMESTEPS = [
    (TimeStepSpec([slice(None, None, -10)]), "Step size must be >= 1"),
]

TESTCASES_WARNING = [
    (0, "Auto output is disabled because period is set to 0 or an empty TimeStepSpec"),
    (TimeStepSpec(), "Auto output is disabled because period is set to 0 or an empty TimeStepSpec"),
]


TESTCASES_INVALID_GET_AS = [
    (10, {}, -0.5, 200, "time_step_size must be positive"),
]


class PICMI_TestAuto(unittest.TestCase):
    def test_auto(self):
        """Test Auto instantiation, validation, and serialization."""
        for period, expected_specs in TESTCASES_VALID:
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
                auto.check()
                pypicongpu_auto = auto.get_as_pypicongpu({}, 0.5, 200)
                self.assertIsInstance(pypicongpu_auto, PyPIConGPUAuto)
                self.assertIsInstance(pypicongpu_auto.period, PyPIConGPUTimeStepSpec)
                serialized = pypicongpu_auto.get_rendering_context()
                self.assertTrue(serialized["typeID"]["auto"])
                self.assertEqual(serialized["data"]["period"]["specs"], expected_specs)
                self.assertEqual(serialized["data"]["png_axis"], [{"axis": "yx"}, {"axis": "yz"}])

        for period, expected_error in TESTCASES_INVALID:
            with self.subTest(period=period, expected_error=expected_error):
                if isinstance(period, str):
                    with self.assertRaises(typeguard.TypeCheckError):
                        Auto(period=period)
                else:
                    with self.assertRaisesRegex((ValueError, TypeError), expected_error):
                        Auto(period=period)

    def test_auto_warning(self):
        """Test warning for disabled Auto output."""
        for period, expected_warning in TESTCASES_WARNING:
            with self.subTest(period=period, expected_warning=expected_warning):
                auto = Auto(period=period)
                with self.assertWarnsRegex(UserWarning, expected_warning):
                    auto.check()

    def test_auto_get_as_pypicongpu(self):
        """Test get_as_pypicongpu with invalid simulation parameters."""
        auto = Auto(period=10)
        for _, _, time_step_size, num_steps, expected_error in TESTCASES_INVALID_GET_AS:
            with self.subTest(time_step_size=time_step_size, num_steps=num_steps, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    auto.get_as_pypicongpu({}, time_step_size, num_steps)

    def test_auto_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        auto = Auto(period=10)
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            auto.get_as_pypicongpu({}, -0.5, 200)


if __name__ == "__main__":
    unittest.main()
