"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import os
import re
import tempfile
from unittest import TestCase

from picongpu import picmi
from picongpu.picmi.diagnostics import ParticleEnergy, TimeStepSpec
from picongpu.picmi.particle_functor import ParticleFilter


class TestParticleEnergy(TestCase):
    @staticmethod
    def __get_species():
        return picmi.Species(name="e", particle_type="electron")

    def __get_energy(self, species):
        return ParticleEnergy(species=species, period=TimeStepSpec[:16:2])

    def test_converts_to_pypicongpu(self):
        converted = self.__get_energy(self.__get_species()).get_as_pypicongpu(time_step_size=1, num_steps=17)

        assert converted.type_particleenergy is True
        assert converted.species.species_name == "e"
        assert converted.species.filter_name == "all"
        assert len(converted.period.specs) == 1
        spec = converted.period.specs[0]
        assert (spec.start, spec.stop, spec.step) == (0, 16, 2)

    def test_filtered_species_uses_functor_name_as_filter(self):
        species = self.__get_species()
        filtered = picmi.FilteredSpecies(
            species=species,
            functor=ParticleFilter(name="positive", functor=lambda p: p.get("momentum")[2] > 0.0),
        )

        converted = self.__get_energy(filtered).get_as_pypicongpu(time_step_size=1, num_steps=17)

        assert converted.species.species_name == "e"
        assert converted.species.filter_name == "positive"


class TestParticleEnergyRenderedNcfg(TestCase):
    @staticmethod
    def __build_sim(diagnostic, species):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[32, 32, 32],
            lower_bound=[0, 0, 0],
            upper_bound=[3.2e-6, 3.2e-6, 3.2e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid, cfl=1.0)
        sim = picmi.Simulation(time_step_size=None, max_steps=17, solver=solver)
        sim.add_species(species, picmi.PseudoRandomLayout(n_macroparticles_per_cell=2))
        sim.diagnostics = [diagnostic]
        return sim

    @staticmethod
    def __read_n_cfg(outdir):
        n_cfg = os.path.join(outdir, "etc", "picongpu", "N.cfg")
        if not os.path.isfile(n_cfg):
            candidates = [
                os.path.join(root, name) for root, _, names in os.walk(outdir) for name in names if name == "N.cfg"
            ]
            assert candidates, f"no rendered N.cfg found under {outdir}"
            n_cfg = candidates[0]
        with open(n_cfg) as cfg:
            return cfg.read()

    def test_unfiltered_renders_period_and_all_filter(self):
        species = picmi.Species(
            name="e",
            particle_type="electron",
            initial_distribution=picmi.UniformDistribution(density=1e20),
        )
        sim = self.__build_sim(ParticleEnergy(species=species, period=TimeStepSpec[:16:2]), species)

        with tempfile.TemporaryDirectory() as outdir:
            sim.write_input_file(outdir, exist_ok=True)
            content = self.__read_n_cfg(outdir)

        assert "--e_energy.period 0:16:2" in content
        assert "--e_energy.filter all" in content

    def test_filtered_renders_period_and_functor_filter(self):
        species = picmi.Species(
            name="e",
            particle_type="electron",
            initial_distribution=picmi.UniformDistribution(density=1e20),
        )
        filtered = picmi.FilteredSpecies(
            species=species,
            functor=ParticleFilter(name="positive", functor=lambda p: p.get("momentum")[2] > 0.0),
        )
        sim = self.__build_sim(ParticleEnergy(species=filtered, period=TimeStepSpec[:16:2]), species)

        with tempfile.TemporaryDirectory() as outdir:
            sim.write_input_file(outdir, exist_ok=True)
            content = self.__read_n_cfg(outdir)

        assert "--e_energy.period 0:16:2" in content
        assert "--e_energy.filter positive" in content

    def test_period_token_format_matches_energy_histogram_style(self):
        species = picmi.Species(
            name="e",
            particle_type="electron",
            initial_distribution=picmi.UniformDistribution(density=1e20),
        )
        sim = self.__build_sim(ParticleEnergy(species=species, period=TimeStepSpec[0:10:5]), species)

        with tempfile.TemporaryDirectory() as outdir:
            sim.write_input_file(outdir, exist_ok=True)
            content = self.__read_n_cfg(outdir)

        match = re.search(r"--e_energy\.period (\S+)", content)
        assert match is not None, "particle-energy period not found in rendered N.cfg"
        assert match.group(1) == "0:10:5"
