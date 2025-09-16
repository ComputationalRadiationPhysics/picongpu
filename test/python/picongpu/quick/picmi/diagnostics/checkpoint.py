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
        {"period": {"specs": [{"start": 0, "stop": 199, "step": 10}]}, "timePeriod": None, "directory": "checkpoints"},
    ),
    (
        {
            "period": TimeStepSpec([5, 10]),
            "timePeriod": 10,
            "restartStep": 100,
            "restartDirectory": "backups",
            "restartFile": "backup",
            "restartChunkSize": 1000,
            "restartLoop": 2,
            "openPMD": {"ext": "h5"},
        },
        {
            "period": {"specs": [{"start": 5, "stop": 6, "step": 1}, {"start": 10, "stop": 11, "step": 1}]},
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

TESTCASES_INVALID = [
    ({"period": None, "timePeriod": None}, "At least one of period or timePeriod must be provided"),
    ({"period": 10, "timePeriod": -5}, "timePeriod must be a non-negative"),
    ({"period": 10, "restartStep": -1}, "restartStep must be non-negative"),
    ({"period": 10, "restartChunkSize": 0}, "restartChunkSize must be positive"),
    ({"period": 10, "restartLoop": -1}, "restartLoop must be non-negative"),
    ({"period": "invalid", "timePeriod": None}, 'argument "period".*did not match any element'),
]

TESTCASES_WARNING = [
    (
        {"period": 0, "timePeriod": 0},
        "Checkpoint is disabled because period is set to 0 or an empty TimeStepSpec and timePeriod is None or 0",
    ),
    (
        {"period": TimeStepSpec([]), "timePeriod": 0},
        "Checkpoint is disabled because period is set to 0 or an empty TimeStepSpec and timePeriod is None or 0",
    ),
]

TESTCASES_INVALID_GET_AS = [
    ({"period": TimeStepSpec([slice(None, None, -10)]), "timePeriod": None}, "Step size must be >= 1"),
    # Skip non-raising cases
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
                        expected = TimeStepSpec([slice(None, None, value)] if value > 0 else [])("steps")
                        self.assertEqual(checkpoint.period.specs, expected.specs)
                    else:
                        self.assertEqual(getattr(checkpoint, key), value)
                checkpoint.check()
                pypicongpu_checkpoint = checkpoint.get_as_pypicongpu({}, 0.5, 200)
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

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                    Checkpoint(**params)

    def test_checkpoint_warning(self):
        """Test warning for disabled Checkpoint."""
        for params, expected_warning in TESTCASES_WARNING:
            with self.subTest(params=params):
                checkpoint = Checkpoint(**params)
                with self.assertWarnsRegex(UserWarning, expected_warning):
                    checkpoint.check()

    def test_checkpoint_invalid_cases(self):
        """Test invalid TimeStepSpec and simulation parameters."""
        for params, *args in TESTCASES_INVALID_GET_AS:
            with self.subTest(params=params, args=args):
                checkpoint = Checkpoint(**params)
                time_step_size, num_steps = args if len(args) == 2 else (0.5, 200)
                expected_error, *skip = args[-1] if len(args) == 2 else "Step size must be >= 1"
                if skip and skip[0]:  # Skip if flagged
                    continue
                with self.assertRaisesRegex(ValueError, expected_error):
                    checkpoint.get_as_pypicongpu({}, time_step_size, num_steps)


if __name__ == "__main__":
    unittest.main()
