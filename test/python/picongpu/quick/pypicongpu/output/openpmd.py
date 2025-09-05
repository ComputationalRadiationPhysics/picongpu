"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import unittest
import typeguard
from picongpu.pypicongpu.output import OpenPMD
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.attribute import Position, Momentum
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.output.openpmd_sources import (
    BoundElectronDensity,
    EnergyDensityCutoff,
    MidCurrentDensityComponent,
)


def create_species():
    species = Species()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class TestOpenPMD(unittest.TestCase):
    def setUp(self):
        self.species = create_species()
        self.period = TimeStepSpec([slice(0, None, 100)])

    # ---------------------------
    # Valid OpenPMD configurations
    # ---------------------------
    def test_openpmd_valid(self):
        """All valid OpenPMD inputs succeed, including minimal config."""
        sources = [
            BoundElectronDensity(species=self.species, filter="species_all"),
            EnergyDensityCutoff(species=self.species, filter="species_all", cutoff_max_energy=1.0),
            MidCurrentDensityComponent(species=self.species, filter="species_all", direction="x"),
        ]

        # --- Full configuration ---
        openpmd = OpenPMD(
            period=self.period, source=sources, file="output_file", ext="bp", infix="NULL", file_writing="create"
        )
        openpmd.check()
        context = openpmd._get_serialized()
        self.assertEqual(len(context["source"]), 3)
        self.assertEqual(context["source"][0]["type"], "boundelectrondensity")
        self.assertEqual(context["source"][0]["filter"], "species_all")
        self.assertEqual(context["source"][1]["type"], "energydensitycutoff")
        self.assertEqual(context["source"][1]["cutoff_max_energy"], 1.0)
        self.assertEqual(context["source"][2]["type"], "midcurrentdensitycomponent")
        self.assertEqual(context["source"][2]["direction"], "x")

        # --- Minimal configuration ---
        minimal = OpenPMD(period=TimeStepSpec([slice(None, None, None)]), source=[], file="output", ext="bp")
        context_min = minimal._get_serialized()
        self.assertEqual(context_min["source"], [])
        self.assertEqual(context_min["file"], "output")
        self.assertEqual(context_min["ext"], "bp")

    # ---------------------------
    # Invalid argument tests
    # ---------------------------
    def test_openpmd_invalid_arguments(self):
        """Invalid OpenPMD arguments raise proper exceptions."""
        sources = [
            BoundElectronDensity(species=self.species, filter="species_all"),
            EnergyDensityCutoff(species=self.species, filter="species_all", cutoff_max_energy=1.0),
            MidCurrentDensityComponent(species=self.species, filter="species_all", direction="x"),
        ]

        # --- OpenPMD argument type errors ---
        invalid_args = {
            "period": ["string", 123],
            "source": ["string", 123, [123]],
            "file": ["", 123],
            "ext": ["txt", 123],
            "file_writing": ["overwrite", 123],
        }

        for arg, values in invalid_args.items():
            for val in values:
                with self.subTest(arg=arg, val=val):
                    kwargs = {
                        "period": self.period,
                        "source": sources,
                        "file": "out",
                        "ext": "bp",
                        "file_writing": "create",
                    }
                    kwargs[arg] = val

                    if arg == "file" and val == "":
                        # empty string triggers ValueError from check()
                        with self.assertRaises(ValueError):
                            OpenPMD(**kwargs)
                    else:
                        # everything else triggers TypeGuard type check
                        with self.assertRaises(typeguard.TypeCheckError):
                            OpenPMD(**kwargs)

        # --- Source-specific validation errors ---
        with self.assertRaisesRegex(ValueError, "must be positive"):
            EnergyDensityCutoff(species=self.species, filter="species_all", cutoff_max_energy=0).check()

        with self.assertRaisesRegex(ValueError, "Direction must be"):
            MidCurrentDensityComponent(species=self.species, filter="species_all", direction="w").check()

        with self.assertRaisesRegex(ValueError, "Filter must be one of"):
            BoundElectronDensity(species=self.species, filter="invalid_filter").check()

        # --- Empty file string triggers ValueError ---
        with self.assertRaisesRegex(ValueError, "file must be a non-empty string"):
            OpenPMD(period=self.period, source=[], file="", ext="bp").check()


if __name__ == "__main__":
    unittest.main()
