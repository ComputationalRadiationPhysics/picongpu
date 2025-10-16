"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec
from picongpu.pypicongpu.output.checkpoint import Checkpoint as PyPIConGPUCheckpoint
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec
import unittest
import typeguard

TESTCASES_VALID = [
    (
        {"period": 10, "timePeriod": None, "directory": "checkpoints"},
        {"period": {"specs": [{"start": 0, "stop": -1, "step": 10}]}, "timePeriod": None, "directory": "checkpoints"},
    ),
    (
        {
            "period": TimeStepSpec(5, 10)("steps"),
            "timePeriod": 10,
            "restartStep": 100,
            "restartDirectory": "backups",
            "restartFile": "backup",
            "restartChunkSize": 1000,
            "restartLoop": 2,
            "openPMD": {"ext": "h5"},
        },
        {
            "period": {"specs": [{"start": 5, "stop": 5, "step": 1}, {"start": 10, "stop": 10, "step": 1}]},
            "timePeriod": 10,
            "restartStep": 100,
            "restartDirectory": "backups",
            "restartFile": "backup",
            "restartChunkSize": 1000,
            "restartLoop": 2,
            "openPMD": {"ext": "h5"},
        },
    ),
]

logic_invalid_cases = [
    ({"period": None, "timePeriod": None}, "At least one of period or timePeriod must be provided and active"),
    ({"period": None, "timePeriod": 0}, "At least one of period or timePeriod must be provided and active"),
    ({"period": 0, "timePeriod": 0}, "At least one of period or timePeriod must be provided and active"),
    (
        {"period": TimeStepSpec()("steps"), "timePeriod": 0},
        "At least one of period or timePeriod must be provided and active",
    ),
    ({"period": 10, "timePeriod": -5}, "timePeriod must be a non-negative"),
    ({"period": 10, "restartStep": -1}, "restartStep must be non-negative"),
    ({"period": 10, "restartChunkSize": 0}, "restartChunkSize must be positive"),
    ({"period": 10, "restartLoop": -1}, "restartLoop must be non-negative"),
]

type_invalid_cases = [
    ({"period": "invalid", "timePeriod": None}, 'argument "period".*did not match any element'),
]

TESTCASES_INVALID_GET_AS = [
    ({"period": TimeStepSpec([slice(None, None, -10)]), "timePeriod": None}, "Step size must be >= 1"),
    ({"period": 10, "timePeriod": None}, -0.5, 200, "time_step_size must be positive", True),
    ({"period": 10, "timePeriod": None}, 0.5, 0, "num_steps must be positive", True),
]


class PICMI_TestCheckpoint(unittest.TestCase):
    def test_checkpoint(self):
        """Test Checkpoint instantiation, validation, and serialization."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params):
                checkpoint = Checkpoint(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = TimeStepSpec(slice(None, None, value))("steps")
                        self.assertEqual(checkpoint.period.specs, expected.specs)
                    else:
                        self.assertEqual(getattr(checkpoint, key), value)
                pypicongpu_checkpoint = checkpoint.get_as_pypicongpu(0.5, 200)
                self.assertIsInstance(pypicongpu_checkpoint, PyPIConGPUCheckpoint)
                self.assertIsInstance(pypicongpu_checkpoint.period, PyPIConGPUTimeStepSpec)
                serialized_data = pypicongpu_checkpoint._get_serialized()
                serialized = {"typeID": {"checkpoint": True}, "data": serialized_data}
                self.assertEqual(serialized["typeID"], {"checkpoint": True})
                for key, value in expected_serialized.items():
                    if key == "period":
                        self.assertEqual(serialized_data["period"]["specs"], value["specs"])
                    elif key in serialized_data:
                        self.assertEqual(serialized_data[key], value)
                    else:
                        self.assertIsNone(value)

        for params, expected_error in logic_invalid_cases:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    Checkpoint(**params)

        for params, expected_error in type_invalid_cases:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex(typeguard.TypeCheckError, expected_error):
                    Checkpoint(**params)

    def test_checkpoint_invalid_cases(self):
        """Test invalid TimeStepSpec and simulation parameters."""
        for params, *args in TESTCASES_INVALID_GET_AS:
            with self.subTest(params=params, args=args):
                checkpoint = Checkpoint(**params)
                time_step_size, num_steps = args if len(args) == 2 else (0.5, 200)
                expected_error, *skip = args[-1] if len(args) == 2 else "Step size must be >= 1"
                if skip and skip[0]:
                    continue
                with self.assertRaisesRegex(ValueError, expected_error):
                    checkpoint.get_as_pypicongpu({}, time_step_size, num_steps)


if __name__ == "__main__":
    unittest.main()
