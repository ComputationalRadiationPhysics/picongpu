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


# Test cases for valid Checkpoint inputs
TESTCASES_VALID = [
    (
        {"period": 10, "timePeriod": None, "directory": "checkpoints", "restart": None, "tryRestart": None},
        {
            "period": {"specs": [{"start": 0, "stop": 199, "step": 10}]},
            "timePeriod": None,
            "directory": "checkpoints",
            "restart": None,
            "tryRestart": None,
        },
    ),
    (
        {"period": 0, "timePeriod": 5, "file": "chkpt", "tryRestart": False, "restart": True},
        {"period": {"specs": []}, "timePeriod": 5, "file": "chkpt", "tryRestart": False, "restart": True},
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
    (
        {"period": TimeStepSpec([slice(-10, None, 1)]), "timePeriod": None},
        {
            "period": {"specs": [{"start": 190, "stop": 199, "step": 1}]},
            "timePeriod": None,
        },  # Adjusted for num_steps=200
    ),
    ({"period": 0, "timePeriod": 0}, {"period": {"specs": []}, "timePeriod": 0}),
]

# Invalid test cases for instantiation
TESTCASES_INVALID = [
    (
        {"period": None, "timePeriod": None},
        "At least one of period or timePeriod must be provided to enable checkpointing",
    ),
    ({"period": 10, "timePeriod": -5}, "timePeriod must be a non-negative integer"),
    ({"period": 10, "restartStep": -1}, "restartStep must be non-negative"),
    ({"period": 10, "restartChunkSize": 0}, "restartChunkSize must be positive"),
    ({"period": 10, "restartLoop": -1}, "restartLoop must be non-negative"),
    (
        {"period": "invalid", "timePeriod": None},
        'argument "period".*did not match any element in the union',
    ),
]

# Invalid test cases for TimeStepSpec with negative steps
TESTCASES_INVALID_TIMESTEPS = [
    ({"period": TimeStepSpec([slice(None, None, -10)]), "timePeriod": None}, "Step size must be >= 1"),
]

# Test cases for warning when checkpoint is disabled
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


class PICMI_TestCheckpoint(unittest.TestCase):
    def test_checkpoint_instantiation(self):
        """Test Checkpoint instantiation and validation."""
        for params, _ in TESTCASES_VALID:
            with self.subTest(params=params):
                checkpoint = Checkpoint(**params)
                for key, value in params.items():
                    if key == "period" and isinstance(value, int):
                        expected = TimeStepSpec([slice(None, None, value)] if value > 0 else [])("steps")
                        self.assertEqual(checkpoint.period.specs, expected.specs)
                    else:
                        self.assertEqual(getattr(checkpoint, key), value)
                if not checkpoint.period.specs and (params.get("timePeriod") is None or params.get("timePeriod") == 0):
                    with self.assertWarnsRegex(
                        UserWarning,
                        "Checkpoint is disabled because period is set to 0 or an empty TimeStepSpec and timePeriod is None or 0",
                    ):
                        checkpoint.check()
                else:
                    checkpoint.check()  # Should not raise or warn

        for params, expected_error in TESTCASES_INVALID:
            with self.subTest(params=params, expected_error=expected_error):
                with self.assertRaisesRegex((ValueError, typeguard.TypeCheckError), expected_error):
                    Checkpoint(**params)

    def test_checkpoint_serialization(self):
        """Test Checkpoint serialization to PyPIConGPUCheckpoint."""
        for params, expected_serialized in TESTCASES_VALID:
            with self.subTest(params=params, expected_serialized=expected_serialized):
                checkpoint = Checkpoint(**params)
                pypicongpu_checkpoint = checkpoint.get_as_pypicongpu(0.5, 200)
                self.assertIsInstance(pypicongpu_checkpoint, PyPIConGPUCheckpoint)
                self.assertIsInstance(pypicongpu_checkpoint.period, PyPIConGPUTimeStepSpec)
                serialized = pypicongpu_checkpoint.get_rendering_context()
                self.assertEqual(serialized["typeID"], {"checkpoint": True})
                serialized_data = serialized["data"]
                for key, value in expected_serialized.items():
                    if key == "period":
                        for i, spec in enumerate(value["specs"]):
                            self.assertEqual(serialized_data["period"]["specs"][i]["start"], spec["start"])
                            self.assertEqual(serialized_data["period"]["specs"][i]["stop"], spec["stop"])
                            self.assertEqual(serialized_data["period"]["specs"][i]["step"], spec["step"])
                    else:
                        self.assertEqual(serialized_data[key], value)

    def test_checkpoint_warning(self):
        """Test warning for disabled Checkpoint."""
        for params, expected_warning in TESTCASES_WARNING:
            with self.subTest(params=params, expected_warning=expected_warning):
                checkpoint = Checkpoint(**params)
                with self.assertWarnsRegex(UserWarning, expected_warning):
                    checkpoint.check()

    def test_checkpoint_invalid_timestepspec(self):
        """Test invalid TimeStepSpec with negative steps."""
        for params, expected_error in TESTCASES_INVALID_TIMESTEPS:
            with self.subTest(params=params, expected_error=expected_error):
                checkpoint = Checkpoint(**params)
                with self.assertRaisesRegex(ValueError, expected_error):
                    checkpoint.get_as_pypicongpu(0.5, 200)

    def test_checkpoint_invalid_simulation_parameters(self):
        """Test invalid simulation parameters in get_as_pypicongpu."""
        checkpoint = Checkpoint(period=10)
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            checkpoint.get_as_pypicongpu(-0.5, 200)
        with self.assertRaisesRegex(ValueError, "time_step_size must be positive"):
            checkpoint.get_as_pypicongpu(0, 200)
        with self.assertRaisesRegex(ValueError, "num_steps must be positive"):
            checkpoint.get_as_pypicongpu(0.5, 0)


if __name__ == "__main__":
    unittest.main()
