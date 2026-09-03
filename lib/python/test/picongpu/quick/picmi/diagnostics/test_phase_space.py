"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

from unittest import TestCase

from picongpu import picmi
from picongpu.picmi import constants
from picongpu.picmi.diagnostics import PhaseSpace, TimeStepSpec
from picongpu.picmi.particle_functor import ParticleFilter


class TestPhaseSpace(TestCase):
    @staticmethod
    def __get_species():
        return picmi.Species(name="e", particle_type="electron")

    def __get_phase_space(self, species, min_momentum_si, max_momentum_si):
        return PhaseSpace(
            species=species,
            period=TimeStepSpec[:16:2],
            spatial_coordinate="y",
            momentum_coordinate="pz",
            min_momentum=min_momentum_si,
            max_momentum=max_momentum_si,
        )

    def test_si_momentum_converted_to_species_mc_for_backend(self):
        # momenta are given in SI (kg*m/s) and must be passed to the backend in m_species*c
        species = self.__get_species()
        mass_si = species.picongpu_get_mass_si()
        momentum_unit_si = mass_si * constants.c

        phase_space = self.__get_phase_space(species, -3.0 * momentum_unit_si, 5.0 * momentum_unit_si)

        converted = phase_space.get_as_pypicongpu(time_step_size=1, num_steps=17)

        assert converted.type_phasespace is True
        assert abs(converted.min_momentum - (-3.0)) < 1e-12
        assert abs(converted.max_momentum - 5.0) < 1e-12

    def test_arbitrary_si_momentum(self):
        # an arbitrary SI momentum value maps to the corresponding multiple of m_species*c
        species = self.__get_species()
        mass_si = species.picongpu_get_mass_si()
        momentum_si = 1.37e-19

        phase_space = self.__get_phase_space(species, 0.0, momentum_si)

        converted = phase_space.get_as_pypicongpu(time_step_size=1, num_steps=17)

        assert abs(converted.max_momentum - momentum_si / (mass_si * constants.c)) < 1e-12

    def test_filtered_species_uses_species_mass(self):
        species = self.__get_species()
        mass_si = species.picongpu_get_mass_si()
        momentum_unit_si = mass_si * constants.c

        filtered = picmi.FilteredSpecies(
            species=species,
            functor=ParticleFilter(name="positive", functor=lambda p: p.get("momentum")[2] > 0.0),
        )

        phase_space = PhaseSpace(
            species=filtered,
            period=TimeStepSpec[:16:2],
            spatial_coordinate="y",
            momentum_coordinate="pz",
            min_momentum=-2.0 * momentum_unit_si,
            max_momentum=2.0 * momentum_unit_si,
        )

        converted = phase_space.get_as_pypicongpu(time_step_size=1, num_steps=17)

        assert abs(converted.min_momentum - (-2.0)) < 1e-12
        assert abs(converted.max_momentum - 2.0) < 1e-12

    def test_species_without_mass_rejected(self):
        species = picmi.Species(name="custom")

        phase_space = self.__get_phase_space(species, 0.0, 1.0)

        with self.assertRaises(ValueError):
            phase_space.get_as_pypicongpu(time_step_size=1, num_steps=17)
