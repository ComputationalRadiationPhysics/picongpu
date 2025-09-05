"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import Checkpoint
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
import unittest
import typeguard


class TestCheckpoint(unittest.TestCase):
    def test_instantiation_and_types(self):
        """Test instantiation, type safety, and valid serialization."""
        # Valid configuration with period
        cp = Checkpoint()
        cp.period = TimeStepSpec([slice(0, None, 100)])
        cp.directory = "checkpoints"
        cp.file = "checkpoint_%T"
        cp.restart = True
        cp.tryRestart = False
        cp.restartStep = 0
        cp.restartDirectory = "restart"
        cp.restartFile = "restart_%T"
        cp.restartChunkSize = 1
        cp.restartLoop = 0
        cp.openPMD = {"backend": "bp"}
        cp.check()
        context = cp.get_rendering_context()
        self.assertTrue(context["typeID"]["checkpoint"])
        self.assertEqual(context["data"]["period"]["specs"][0]["step"], 100)
        self.assertIsNone(context["data"].get("timePeriod"))

        # Valid configuration with timePeriod
        cp = Checkpoint()
        cp.timePeriod = 100
        context = cp.get_rendering_context()
        self.assertTrue(context["typeID"]["checkpoint"])
        self.assertEqual(context["data"]["timePeriod"], 100)
        self.assertIsNone(context["data"].get("period"))

        # Type safety
        invalid_types = {
            "period": ["string", 1],
            "timePeriod": ["string", 1.5],
            "directory": [1, []],
            "file": [1, []],
            "restart": ["string", 1],
            "tryRestart": ["string", 1],
            "restartStep": ["string", 1.5],
            "restartDirectory": [1, []],
            "restartFile": [1, []],
            "restartChunkSize": ["string", 1.5],
            "restartLoop": ["string", 1.5],
            "openPMD": ["string", 1],
        }
        for attr, invalid_values in invalid_types.items():
            for value in invalid_values:
                with self.subTest(attr=attr, value=value):
                    cp = Checkpoint()
                    with self.assertRaises(typeguard.TypeCheckError):
                        setattr(cp, attr, value)

    def test_rendering_and_validation(self):
        """Test serialization output, validation errors, and edge cases."""
        # Valid full serialization
        cp = Checkpoint()
        cp.period = TimeStepSpec([slice(0, None, 100)])
        cp.timePeriod = 100
        cp.directory = "checkpoints"
        cp.file = "checkpoint_%T"
        cp.restart = True
        cp.tryRestart = False
        cp.restartStep = 0
        cp.restartDirectory = "restart"
        cp.restartFile = "restart_%T"
        cp.restartChunkSize = 1
        cp.restartLoop = 0
        cp.openPMD = {"backend": "bp"}
        context = cp.get_rendering_context()
        self.assertTrue(context["typeID"]["checkpoint"])
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
        self.assertEqual(context["timePeriod"], 100)
        self.assertEqual(context["directory"], "checkpoints")
        self.assertEqual(context["file"], "checkpoint_%T")
        self.assertTrue(context["restart"])
        self.assertFalse(context["tryRestart"])
        self.assertEqual(context["restartStep"], 0)
        self.assertEqual(context["restartDirectory"], "restart")
        self.assertEqual(context["restartFile"], "restart_%T")
        self.assertEqual(context["restartChunkSize"], 1)
        self.assertEqual(context["restartLoop"], 0)
        self.assertEqual(context["openPMD"], {"backend": "bp"})

        # Validation errors
        cp = Checkpoint()
        with self.assertRaisesRegex(ValueError, "At least one of period or timePeriod must be provided"):
            cp.get_rendering_context()

        cp = Checkpoint()
        cp.timePeriod = -1
        with self.assertRaisesRegex(ValueError, "timePeriod must be non-negative"):
            cp.get_rendering_context()

        cp = Checkpoint()
        cp.timePeriod = 100
        cp.restartStep = -1
        with self.assertRaisesRegex(ValueError, "restartStep must be non-negative"):
            cp.get_rendering_context()

        cp = Checkpoint()
        cp.timePeriod = 100
        cp.restartChunkSize = 0
        with self.assertRaisesRegex(ValueError, "restartChunkSize must be positive"):
            cp.get_rendering_context()

        cp = Checkpoint()
        cp.timePeriod = 100
        cp.restartLoop = -1
        with self.assertRaisesRegex(ValueError, "restartLoop must be non-negative"):
            cp.get_rendering_context()

        cp = Checkpoint()
        cp.period = TimeStepSpec([slice(0, None, -1)])
        with self.assertRaisesRegex(ValueError, "Step size must be >= 1"):
            cp.get_rendering_context()


if __name__ == "__main__":
    unittest.main()
