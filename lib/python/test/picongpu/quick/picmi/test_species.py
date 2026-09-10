"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

import pytest
from picongpu.picmi.interaction.ionization.fieldionization import ADK, BSI
from picongpu.picmi.species import Species
from picongpu.picmi.species_requirements import RequirementConflict, SetChargeStateOperation, run_construction
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState
from picongpu.pypicongpu.species.species import Pusher, Shape


def species(**kwargs):
    return Species(name="electron", particle_type="electron", **kwargs)


class TestSpeciesShapeAndMethod(TestCase):
    def _assert_converts(self, shape, method, expected_shape, expected_pusher):
        pypicongpu_species = species(particle_shape=shape, method=method).get_as_pypicongpu()
        self.assertIs(pypicongpu_species.shape, expected_shape)
        self.assertIs(pypicongpu_species.pusher, expected_pusher)

    def test_standard_shapes(self):
        for shape, expected in {
            "NGP": Shape.NGP,
            "linear": Shape.linear,
            "quadratic": Shape.quadratic,
            "cubic": Shape.cubic,
        }.items():
            with self.subTest(shape=shape):
                self._assert_converts(shape, "Boris", expected, Pusher.Boris)

    def test_standard_pusher_methods(self):
        for method, expected in {
            "Boris": Pusher.Boris,
            "Vay": Pusher.Vay,
            "Higuera-Cary": Pusher.Higuera,
            "free-streaming": Pusher.Free,
            "LLRK4": Pusher.ReducedLandauLifshitz,
        }.items():
            with self.subTest(method=method):
                self._assert_converts("quadratic", method, Shape.quadratic, expected)

    def test_picongpu_extensions(self):
        self._assert_converts("other:quartic", "other:Acceleration", Shape.quartic, Pusher.Acceleration)
        self._assert_converts("other:counter", "other:Photon", Shape.counter, Pusher.Photon)
        self._assert_converts("other:quartic", "other:Probe", Shape.quartic, Pusher.Probe)
        self._assert_converts("other:quartic", "other:Axel", Shape.quartic, Pusher.Axel)

    def test_method_explicitly_set_does_not_crash(self):
        # Regression: picmistandard's _validate_method used to raise
        # AttributeError for any explicit (non-default) method.
        pypicongpu_species = species(method="Vay").get_as_pypicongpu()
        self.assertIs(pypicongpu_species.pusher, Pusher.Vay)

    def test_standard_unimplemented_method_accepted_but_rejected_at_conversion(self):
        # "Li" is a standard method that PIConGPU does not implement.
        construct = species(method="Li")
        self.assertEqual(construct.method, "Li")
        with self.assertRaises(ValueError, msg="Li must be rejected at conversion time"):
            construct.get_as_pypicongpu()

    def test_unknown_other_accepted_but_rejected_at_conversion(self):
        for value in ("other:SomeUnknownPusher", "other:SomeUnknownShape"):
            with self.subTest(value=value):
                field = "method" if value.startswith("other:SomeUnknownPusher") else "particle_shape"
                construct = species(**{field: value})
                with self.assertRaises(ValueError, msg=f"{value} must be rejected at conversion time"):
                    construct.get_as_pypicongpu()


class TestSpeciesNameDefault(TestCase):
    def test_name_defaults_from_particle_type(self):
        s = Species(particle_type="electron")
        self.assertEqual(s.name, "electron")

    def test_name_default_survives_conversion(self):
        # A None name used to slip through to pypicongpu and later surface as a
        # ValidationError at get_as_pypicongpu() (name: str rejects None); it must
        # now be a proper, C++-compatible name after conversion.
        self.assertEqual(Species(particle_type="electron").get_as_pypicongpu().name, "electron")
        self.assertEqual(Species(particle_type="H").name, "H")

    def test_explicit_name_is_kept(self):
        self.assertEqual(Species(particle_type="electron", name="my_e").name, "my_e")

    def test_explicit_none_name_falls_back(self):
        self.assertEqual(Species(particle_type="electron", name=None).name, "electron")

    def test_no_name_and_no_particle_type_raises(self):
        with self.assertRaises(ValueError):
            Species()

    def test_name_defaults_via_model_validate(self):
        # The before-model validator must also apply to the dict-only validation path.
        self.assertEqual(Species.model_validate({"particle_type": "electron"}).name, "electron")

    def test_other_particle_type_requires_explicit_name(self):
        # "other:" particle types are not valid C++ identifiers, so they must not be
        # auto-used as the species name; an explicit name is required.
        with self.assertRaises(ValueError):
            Species(particle_type="other:MyType")
        self.assertEqual(Species(particle_type="other:MyType", name="e").name, "e")

    def test_other_particle_type_defaults_via_model_validate(self):
        with self.assertRaises(ValueError):
            Species.model_validate({"particle_type": "other:MyType"})

    def test_name_assignment_none_not_redefaulted(self):
        # A before-model validator does not run on attribute assignment, so an
        # explicit `name = None` is not re-defaulted from particle_type; it only
        # surfaces as a ValidationError later, at pypicongpu conversion.
        s = Species(particle_type="electron", name="foo")
        s.name = None
        self.assertIsNone(s.name)
        with self.assertRaises(ValueError):
            s.get_as_pypicongpu()


def unique_in(elements, collection):
    collection = list(collection)
    return (collection.count(e) == 1 for e in elements)


class TestSpeciesRequirementResolution(TestCase):
    def test_deduplicate_attributes(self):
        species = Species(name="dummy")
        requirements = [Weighting()]
        species.register_requirements(2 * requirements)
        assert all(unique_in(requirements, species.get_as_pypicongpu().attributes))

    def test_deduplicate_delayed_construction(self):
        species = Species(name="dummy", particle_type="H", charge_state=1)
        requirements = [SetChargeStateOperation(species)]
        species.register_requirements(2 * requirements)
        assert all(unique_in(requirements, species.get_operation_requirements()))

    def test_conflicting_constants(self):
        species = Species(name="dummy")
        requirements = [Mass(mass_si=1.0), Mass(mass_si=2.0)]
        with pytest.raises(RequirementConflict):
            # Not yet decided which one should raise, but one of them definitely will.
            species.register_requirements(requirements)
            species.get_as_pypicongpu()

    def test_ionization(self):
        ion = Species(name="ion", particle_type="H", charge_state=1)
        electron = Species(name="electron", particle_type="electron")
        # These all register requirements:
        ionizations = [
            # Not great: Production code would use the enums not their integer represenation.
            ADK(ion_species=ion, ionization_electron_species=electron, ADK_variant=0, ionization_current=None),
            BSI(ion_species=ion, ionization_electron_species=electron, BSI_extensions=[0], ionization_current=None),
        ]

        # Ionization makes the ion depend on the electron species.
        # This is important for rendering the corresponding C++ header,
        # so the electron species gets defined before the ion species.
        assert electron < ion

        set_charge_state_op = [
            run_construction(op) for op in ion.get_operation_requirements() if op.metadata.Type == SetChargeState
        ][0]
        assert set_charge_state_op.charge_state == ion.charge_state
        assert len(ion.get_as_pypicongpu().constants.ground_state_ionization.ionization_model_list) == len(ionizations)
