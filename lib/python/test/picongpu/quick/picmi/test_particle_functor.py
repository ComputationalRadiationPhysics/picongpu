"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

from sympy import Symbol, sqrt

from picongpu import picmi
from picongpu.picmi import ParticleFilter, ParticleFunctor
from picongpu.picmi.particle_functor.particle_functor import MacroParticle, PhysicalParticle
from picongpu.picmi.particle_functor.unit_dimension import UnitDimension
from picongpu.pypicongpu.species.attribute.boundelectrons import BoundElectrons
from picongpu.pypicongpu.species.attribute.momentum import Momentum
from picongpu.pypicongpu.species.attribute.momentum_prev_1 import MomentumPrev1
from picongpu.pypicongpu.species.attribute.weighting import Weighting


def render(functor, mode="DerivedField"):
    return functor.get_as_pypicongpu(mode)


class TestRequiredAttributes(TestCase):
    """Attributes a functor asks ``Species.register_requirements`` to register."""

    def test_momentum_requires_momentum_attribute(self):
        @ParticleFunctor
        def momentum(particle: MacroParticle):
            return particle.get("momentum")[0]

        self.assertEqual(momentum.get_required_attributes(), [Momentum()])

    def test_momentum_prev_1_requires_its_attribute(self):
        @ParticleFunctor
        def damped(particle: MacroParticle):
            return particle.get("momentumPrev1")

        self.assertEqual(damped.get_required_attributes(), [MomentumPrev1()])

    def test_mass_and_charge_need_no_attribute(self):
        @ParticleFunctor
        def ratio(particle: MacroParticle):
            return particle.get("mass") / particle.get("charge")

        self.assertEqual(ratio.get_required_attributes(), [])

    def test_charge_state_requires_bound_electrons(self):
        @ParticleFunctor
        def state(particle: MacroParticle):
            return particle.get("charge_state")

        self.assertEqual(state.get_required_attributes(), [BoundElectrons()])

    def test_weighting_requires_weighting_attribute(self):
        @ParticleFunctor
        def weight(particle: MacroParticle):
            return particle.get("weighting")

        self.assertEqual(weight.get_required_attributes(), [Weighting()])

    def test_attribute_free_functor_requires_nothing(self):
        @ParticleFunctor
        def nothing(particle: MacroParticle) -> float:
            return 1.0

        self.assertEqual(nothing.get_required_attributes(), [])


class TestUnitDerivation(TestCase):
    """getUnit / getUnitDimension auto-derivation (QA #4/#5)."""

    def _get_unit(self, unit_dimension, unit_factor=None):
        @ParticleFunctor(unit_dimension=unit_dimension, unit_factor=unit_factor)
        def f(particle: MacroParticle) -> float:
            return 1.0

        return render(f).get_unit_cpp

    def _unit_dimension_cpp(self, unit_dimension):
        @ParticleFunctor(unit_dimension=unit_dimension)
        def f(particle: MacroParticle) -> float:
            return 1.0

        return render(f).unit_dimension_cpp

    def test_mass(self):
        self.assertEqual(self._get_unit(UnitDimension(M=1)), "sim.unit.mass()")

    def test_momentum(self):
        self.assertEqual(
            self._get_unit(UnitDimension(L=1, M=1, T=-1)),
            "(sim.unit.length() * sim.unit.mass()) / (sim.unit.time())",
        )

    def test_energy(self):
        self.assertEqual(
            self._get_unit(UnitDimension(L=2, M=1, T=-2)),
            "(sim.unit.length() * sim.unit.length() * sim.unit.mass()) / (sim.unit.time() * sim.unit.time())",
        )

    def test_charge(self):
        self.assertEqual(self._get_unit(UnitDimension(I=1)), "(sim.unit.charge()) / (sim.unit.time())")

    def test_dimensionless(self):
        self.assertEqual(self._get_unit(UnitDimension()), "1.")

    def test_unit_factor_number_overrides_auto_derivation(self):
        # A number is implicitly converted (rendered through the PMAccPrinter).
        self.assertEqual(self._get_unit(UnitDimension(M=1), unit_factor=1.0e6), "1000000.0")

    def test_unit_factor_callable_overrides_auto_derivation(self):
        # A no-argument callable returning a sympy expression is rendered through PMAccPrinter.
        self.assertEqual(self._get_unit(UnitDimension(M=1), unit_factor=lambda: sqrt(2)), "pmacc::math::sqrt(2)")

    def test_unit_factor_sympy_symbol_renders_verbatim(self):
        self.assertEqual(self._get_unit(UnitDimension(N=1), unit_factor=Symbol("sim.unit.mass()")), "sim.unit.mass()")

    def test_unit_factor_rejects_cpp_string(self):
        # No C++ strings in the interface: a raw code string must be rejected.
        with self.assertRaises(ValueError):
            self._get_unit(UnitDimension(M=1), unit_factor="my_factor()")

    def test_unit_dimension_rendered_as_seven_vector(self):
        self.assertEqual(
            self._unit_dimension_cpp(UnitDimension(L=2, M=1, T=-2)), "{2.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0}"
        )

    def test_fractional_exponent_raises(self):
        @ParticleFunctor(unit_dimension=UnitDimension(L=0.5))
        def f(particle: MacroParticle) -> float:
            return 1.0

        # A non-integer 7-vector cannot be turned into a numeric getUnit(); it must be
        # rejected (not silently rounded to a plausible-looking but wrong scale).
        with self.assertRaises(ValueError):
            render(f)

    def test_non_base_unit_component_raises(self):
        @ParticleFunctor(unit_dimension=UnitDimension(N=1))
        def f(particle: MacroParticle) -> float:
            return 1.0

        with self.assertRaises(ValueError):
            render(f)


class TestPhysicalParticleScaling(TestCase):
    """Default (non-manual) PhysicalParticle rescaling must not crash (QA #2)."""

    def _expression(self, functor):
        return render(functor).functor_expression

    def test_physical_momentum_is_identity(self):
        @ParticleFunctor
        def momentum(particle: PhysicalParticle):
            return particle.get("momentum")[0]

        self.assertEqual(self._expression(momentum), "px")

    def test_physical_velocity_is_identity(self):
        @ParticleFunctor
        def velocity(particle: PhysicalParticle):
            return particle.get("velocity")[0]

        self.assertEqual(self._expression(velocity), "vx")

    def test_physical_position_is_identity(self):
        @ParticleFunctor
        def position(particle: PhysicalParticle):
            return particle.get("position", origin="cell")[0]

        self.assertEqual(self._expression(position), "xc_cell_cell")

    def test_physical_damped_weighting_is_identity(self):
        @ParticleFunctor
        def damped(particle: PhysicalParticle):
            return particle.get("damped_weighting")

        self.assertEqual(self._expression(damped), "damped_weighting")

    def test_physical_mass_is_rescaled(self):
        @ParticleFunctor
        def mass(particle: PhysicalParticle):
            return particle.get("mass")

        self.assertEqual(self._expression(mass), "mass/weighting")

    def test_physical_charge_is_rescaled(self):
        @ParticleFunctor
        def charge(particle: PhysicalParticle):
            return particle.get("charge")

        self.assertEqual(self._expression(charge), "charge/weighting")


class TestFilteredDerivedFieldRequirementRegistration(TestCase):
    """A filter inside a ``DerivedFieldDump`` must register its attributes.

    ``DerivedFieldDump`` is handled via the openPMD path, which reads the species
    by name and never converts the ``FilteredSpecies`` wrapper; the filter's
    accessed attributes still have to reach the owning species (regression for
    the ``momentumPrev1`` gap).
    """

    def _converted_species(self, diagnostic):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[8, 8, 8],
            lower_bound=[0, 0, 0],
            upper_bound=[8, 8, 8],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"],
        )
        sim = picmi.Simulation(
            time_step_size=1.0,
            max_steps=4,
            solver=picmi.ElectromagneticSolver(method="Yee", grid=grid),
        )
        species = picmi.Species(particle_type="electron")
        sim.add_species(species, None)
        sim.add_diagnostic(diagnostic(species))
        return sim.get_as_pypicongpu().species[0]

    def test_filter_attributes_are_registered(self):
        from picongpu.picmi.diagnostics import DerivedFieldDump
        from picongpu.picmi.particle_functor import FilteredSpecies

        @ParticleFilter
        def fprev(particle: MacroParticle) -> bool:
            return particle.get("momentumPrev1")[0] > 0

        @ParticleFunctor
        def plain(particle: MacroParticle):
            return particle.get("gamma")

        converted = self._converted_species(
            lambda s: DerivedFieldDump(species=FilteredSpecies(species=s, functor=fprev), functor=plain)
        )
        self.assertIn(MomentumPrev1(), converted.attributes)


class TestParticleClassResolution(TestCase):
    """First-argument annotation resolution (QA #3)."""

    def test_explicit_macro(self):
        @ParticleFunctor
        def f(particle: MacroParticle):
            return particle.get("mass")

        self.assertIs(f._particle_class(), MacroParticle)

    def test_explicit_physical(self):
        @ParticleFunctor
        def f(particle: PhysicalParticle):
            return particle.get("mass")

        self.assertIs(f._particle_class(), PhysicalParticle)

    def test_string_forward_ref_resolves_to_physical(self):
        @ParticleFunctor
        def f(particle: "PhysicalParticle"):
            return particle.get("mass")

        self.assertIs(f._particle_class(), PhysicalParticle)

    def test_string_forward_ref_allows_scaling(self):
        @ParticleFunctor(scales_with_weighting=1)
        def f(particle: "PhysicalParticle"):
            return particle.get("mass")

        # A string-annotated PhysicalParticle must NOT be rejected as non-Physical.
        self.assertIs(f._particle_class(), PhysicalParticle)
        # And it must render without error.
        render(f)

    def test_no_annotation_defaults_to_macro(self):
        @ParticleFunctor
        def f(particle):
            return particle.get("mass")

        self.assertIs(f._particle_class(), MacroParticle)

    def test_zero_arg_functor_raises_clear_error(self):
        with self.assertRaises(TypeError):

            @ParticleFunctor
            def f():
                return 1

    def test_scaling_on_non_physical_raises(self):
        with self.assertRaises(TypeError):

            @ParticleFunctor(scales_with_weighting=1)
            def f(particle: MacroParticle):
                return particle.get("mass")


if __name__ == "__main__":
    import unittest

    unittest.main()
