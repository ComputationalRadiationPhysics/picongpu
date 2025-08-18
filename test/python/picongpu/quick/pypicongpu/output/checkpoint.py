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
from picongpu.picmi.diagnostics import TimeStepSpec as PicmiTimeStepSpec


class TestCheckpoint(unittest.TestCase):
    def test_empty(self):
        """Empty or incomplete configurations are handled correctly."""
        cp = Checkpoint()
        # Neither period nor timePeriod set
        with self.assertRaisesRegex(ValueError, "At least one of period or timePeriod must be provided"):
            cp._get_serialized()

        # Set period but test negative timePeriod
        cp.period = TimeStepSpec([slice(0, None, 100)])
        cp.timePeriod = -1
        with self.assertRaisesRegex(ValueError, "timePeriod must be non-negative"):
            cp._get_serialized()

        # Set valid minimal configuration with period
        cp = Checkpoint()
        cp.period = TimeStepSpec([slice(0, None, 100)])
        serialized = cp.get_rendering_context()
        self.assertTrue(serialized["typeID"]["checkpoint"])
        self.assertEqual(serialized["data"]["period"]["specs"][0]["step"], 100)
        self.assertIsNone(serialized["data"]["timePeriod"])

        # Set valid minimal configuration with timePeriod
        cp = Checkpoint()
        cp.timePeriod = 100
        serialized = cp.get_rendering_context()
        self.assertTrue(serialized["typeID"]["checkpoint"])
        self.assertEqual(serialized["data"]["timePeriod"], 100)
        self.assertIsNone(serialized["data"]["period"])

    def test_types(self):
        """Type safety is ensured for all attributes."""
        cp = Checkpoint()

        # Invalid period
        invalid_periods = [1, "string", [], {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.period = invalid

        # Invalid timePeriod
        invalid_timePeriods = ["string", 1.5, []]
        for invalid in invalid_timePeriods:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.timePeriod = invalid

        # Invalid directory
        invalid_directories = [1, 1.5, []]
        for invalid in invalid_directories:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.directory = invalid

        # Invalid file
        invalid_files = [1, 1.5, []]
        for invalid in invalid_files:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.file = invalid

        # Invalid restart
        invalid_restarts = [1, "string", []]
        for invalid in invalid_restarts:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.restart = invalid

        # Invalid tryRestart
        invalid_tryRestarts = [1, "string", []]
        for invalid in invalid_tryRestarts:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.tryRestart = invalid

        # Invalid restartStep
        invalid_restartSteps = ["string", 1.5, []]
        for invalid in invalid_restartSteps:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.restartStep = invalid

        # Invalid restartDirectory
        invalid_restartDirectories = [1, 1.5, []]
        for invalid in invalid_restartDirectories:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.restartDirectory = invalid

        # Invalid restartFile
        invalid_restartFiles = [1, 1.5, []]
        for invalid in invalid_restartFiles:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.restartFile = invalid

        # Invalid restartChunkSize
        invalid_restartChunkSizes = ["string", 1.5, []]
        for invalid in invalid_restartChunkSizes:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.restartChunkSize = invalid

        # Invalid restartLoop
        invalid_restartLoops = ["string", 1.5, []]
        for invalid in invalid_restartLoops:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.restartLoop = invalid

        # Invalid openPMD
        invalid_openPMDs = [1, "string", []]
        for invalid in invalid_openPMDs:
            with self.assertRaises(typeguard.TypeCheckError):
                cp.openPMD = invalid

        # Valid configuration
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
        cp.get_rendering_context()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
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

        # Test negative index resolution
        cp = Checkpoint()
        cp.period = PicmiTimeStepSpec([slice(-10, None, 1)])
        context = cp.get_rendering_context()
        self.assertEqual(context["data"]["period"]["specs"][0]["start"], 190)
        self.assertEqual(context["data"]["period"]["specs"][0]["stop"], 199)

        # Test integer period
        cp = Checkpoint()
        cp.period = PicmiTimeStepSpec(10)
        context = cp.get_rendering_context()
        self.assertEqual(context["data"]["period"]["specs"][0]["start"], 0)
        self.assertEqual(context["data"]["period"]["specs"][0]["stop"], 199)
        self.assertEqual(context["data"]["period"]["specs"][0]["step"], 10)

        # Unset period and timePeriod should fail
        cp = Checkpoint()
        with self.assertRaisesRegex(ValueError, "At least one of period or timePeriod must be provided"):
            cp.get_rendering_context()

    def test_validation(self):
        """Constraints on parameters are enforced."""
        cp = Checkpoint()

        # Neither period nor timePeriod set
        with self.assertRaisesRegex(ValueError, "At least one of period or timePeriod must be provided"):
            cp.check()

        # Negative timePeriod
        cp.timePeriod = -1
        with self.assertRaisesRegex(ValueError, "timePeriod must be non-negative"):
            cp.check()

        # Negative restartStep
        cp = Checkpoint()
        cp.timePeriod = 100
        cp.restartStep = -1
        with self.assertRaisesRegex(ValueError, "restartStep must be non-negative"):
            cp.check()

        # Non-positive restartChunkSize
        cp = Checkpoint()
        cp.timePeriod = 100
        cp.restartChunkSize = 0
        with self.assertRaisesRegex(ValueError, "restartChunkSize must be positive"):
            cp.check()

        # Negative restartLoop
        cp = Checkpoint()
        cp.timePeriod = 100
        cp.restartLoop = -1
        with self.assertRaisesRegex(ValueError, "restartLoop must be non-negative"):
            cp.check()

        # Invalid TimeStepSpec
        cp = Checkpoint()
        cp.period = PicmiTimeStepSpec([slice(0, None, -1)])
        with self.assertRaisesRegex(ValueError, "Step size must be >= 1"):
            cp.get_rendering_context()

        # Valid configuration
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
        cp.check()  # Should succeed
        serialized = cp.get_rendering_context()["data"]
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)
        self.assertEqual(serialized["timePeriod"], 100)


if __name__ == "__main__":
    unittest.main()
