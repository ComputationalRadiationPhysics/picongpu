"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Tests for step-dependent (composite) particle pushers
(:class:`~picongpu.picmi.species.CompositePusher`).
"""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest
from picongpu import picmi
from picongpu.picmi.diagnostics import TS
from picongpu.picmi.species import CompositePusher, Species


def _grid():
    return picmi.Cartesian3DGrid(
        number_of_cells=[8, 8, 8],
        lower_bound=[0, 0, 0],
        upper_bound=[8, 8, 8],
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )


def _sim(max_steps=4, **kwargs):
    return picmi.Simulation(
        time_step_size=1.0,
        max_steps=max_steps,
        solver=picmi.ElectromagneticSolver(method="Yee", grid=_grid()),
        **kwargs,
    )


def _converted_species(sim, species):
    sim.add_species(species, None)
    return sim.get_as_pypicongpu().species[0]


class TestCompositePusherModel(TestCase):
    def test_first_match_expansion(self):
        # The documented example: Boris wins at 0, 3, 6, 9, 12.
        pusher = CompositePusher(
            {
                TS[::3]("steps"): "Boris",
                TS[:5]("steps"): "Vay",
                TS[6:]("steps"): "free-streaming",
            }
        )
        pusher.validate(13)
        entries = pusher._resolved_entries(13)
        step_to_name = {
            step: next(
                name
                for (slices, _official_name), name in zip(entries, (n for _, n in pusher._items))
                if any(s[0] <= step and (s[1] == -1 or step <= s[1]) and (step - s[0]) % s[2] == 0 for s in slices)
            )
            for step in range(13)
        }
        self.assertEqual(
            step_to_name,
            {
                0: "Boris",
                1: "Vay",
                2: "Vay",
                3: "Boris",
                4: "Vay",
                5: "Vay",
                6: "Boris",
                7: "free-streaming",
                8: "free-streaming",
                9: "Boris",
                10: "free-streaming",
                11: "free-streaming",
                12: "Boris",
            },
        )
        self.assertEqual([name for _, name in pusher._items], ["Boris", "Vay", "free-streaming"])

    def test_scalar_method_still_works(self):
        converted = _converted_species(_sim(), Species(particle_type="electron", method="Boris"))
        self.assertEqual(converted.pusher.value, "Boris")

    def test_higuera_renders_valid_cpp_name(self):
        # The PICMI-facing name "Higuera-Cary" is not a valid C++ identifier;
        # the emitted type must use the struct name "HigueraCary".
        converted = _converted_species(_sim(), Species(particle_type="electron", method="Higuera-Cary"))
        self.assertEqual(converted.pusher_cpp, "particles::pusher::HigueraCary")
        self.assertNotIn("Higuera-Cary", converted.pusher_cpp)


class TestCompositePusherRendering(TestCase):
    def _composite(self):
        return CompositePusher(
            {
                TS[::3]("steps"): "Boris",
                TS[:5]("steps"): "Vay",
                TS[6:]("steps"): "free-streaming",
            }
        )

    def test_right_nested_composite(self):
        converted = _converted_species(
            _sim(max_steps=13),
            Species(particle_type="electron", method=self._composite()),
        )
        self.assertIn("particles::pusher::Composite<", converted.pusher_cpp)
        # right-nested: Boris outermost, Free innermost
        self.assertTrue(converted.pusher_cpp.startswith("particles::pusher::Composite<particles::pusher::Boris,"))
        self.assertIn(
            "particles::pusher::Composite<particles::pusher::Vay, particles::pusher::Free,", converted.pusher_cpp
        )
        # two generated activation functors (N-1 for N=3 stages)
        self.assertEqual(converted.pusher_activation_declaration.count("struct "), 2)
        # generated names follow the <name>_<uuid hex> convention
        self.assertIn("PusherActivation_", converted.pusher_activation_declaration)
        # standalone functor must not derive from the composite base
        self.assertNotIn("particlePusherComposite::Push", converted.pusher_activation_declaration)

    def test_single_entry_is_folded_to_scalar(self):
        converted = _converted_species(
            _sim(max_steps=5),
            Species(particle_type="electron", method=CompositePusher({TS[:]("steps"): "Boris"})),
        )
        self.assertEqual(converted.pusher_cpp, "particles::pusher::Boris")
        self.assertEqual(converted.pusher_activation_declaration, "")

    def test_schedule_is_rendered_into_param_file(self):
        sim = _sim(max_steps=13)
        sim.add_species(Species(particle_type="electron", method=self._composite()), None)
        with TemporaryDirectory() as tmp:
            sim.write_input_file(Path(tmp) / "setup")
            content = (Path(tmp) / "setup" / "include" / "picongpu" / "param" / "speciesDefinition.param").read_text()
        self.assertIn("particles::pusher::Composite<", content)
        self.assertIn("struct PusherActivation_", content)
        self.assertIn("currentStep", content)


class TestCompositePusherValidation(TestCase):
    def test_unmapped_step_is_hard_error(self):
        # step 0 is unclaimed (spec starts at 1); must be rejected.
        pusher = CompositePusher({TS[1:]("steps"): "Boris"})
        with self.assertRaises(ValueError):
            pusher.validate(4)

    def test_single_entry_allowed(self):
        CompositePusher({TS[:]("steps"): "Boris"}).validate(4)

    def test_seconds_unit_key_is_rejected(self):
        pusher = CompositePusher({TS[:]("seconds"): "Boris"})
        with self.assertRaises(ValueError):
            pusher.validate(4)

    def test_adjacent_end_overlap_warns_with_fix(self):
        pusher = CompositePusher({TS[:5]("steps"): "Boris", TS[5:]("steps"): "Vay"})
        with pytest.warns(UserWarning, match="make them disjoint"):
            pusher.validate(8)

    def test_disjoint_adjacent_does_not_warn(self):
        pusher = CompositePusher({TS[:4]("steps"): "Boris", TS[5:]("steps"): "Vay"})
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            pusher.validate(8)
        self.assertEqual(len(record), 0)

    def test_non_end_overlap_is_silent(self):
        # Both claim step 2, but it is not a shared interval end.
        pusher = CompositePusher({TS[::2]("steps"): "Boris", TS[:]("steps"): "Vay"})
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            pusher.validate(6)
        self.assertEqual(len(record), 0)

    def test_interleaved_periodic_schedule_covers_everything(self):
        pusher = CompositePusher({TS[::2]("steps"): "Boris", TS[1::2]("steps"): "Vay"})
        pusher.validate(10)

    def test_composite_is_rejected_on_scalar_method_path(self):
        # A plain scalar method cannot be a CompositePusher (guarded by the type/tag).
        with self.assertRaises(Exception):
            Species(particle_type="electron", method="CompositePusher")

    def test_unknown_pusher_name_in_composite_is_rejected(self):
        with self.assertRaises(ValueError):
            CompositePusher({TS[:]("steps"): "other:SomeUnknownPusher"}).validate(4)
