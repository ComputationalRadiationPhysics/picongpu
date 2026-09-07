"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import json
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from picongpu import picmi
from picongpu.picmi.diagnostics import Binning, BinningAxis, BinSpec, TimeStepSpec
from picongpu.picmi.particle_functor import ParticleFunctor


class TestBinningParticleRegion(TestCase):
    @staticmethod
    def __get_grid():
        return picmi.Cartesian3DGrid(
            number_of_cells=[16, 16, 16],
            lower_bound=[0, 0, 0],
            upper_bound=[1e-6, 1e-6, 1e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
        )

    def __get_binning(self, name, species, particle_region=None, period=None):
        axis = BinningAxis(
            functor=ParticleFunctor(name="dummy", functor=lambda x: 0.0, return_type=float),
            bin_spec=BinSpec(kind="linear", start=-0.5, stop=0.5, nsteps=1),
        )
        kwargs = {}
        if particle_region is not None:
            kwargs["particle_region"] = particle_region
        if period is not None:
            kwargs["period"] = period
        return Binning(
            name=f"{name}_binning",
            deposition_functor=ParticleFunctor(name="unit_count", functor=lambda x: 1.0, return_type=float),
            axes=[axis],
            species=species,
            **kwargs,
        )

    def __get_simulation(self, diagnostics):
        solver = picmi.ElectromagneticSolver(method="Yee", grid=self.__get_grid(), cfl=1.0)
        sim = picmi.Simulation(time_step_size=None, max_steps=16, solver=solver)
        electrons = picmi.Species(
            name="e",
            particle_type="electron",
            initial_distribution=picmi.UniformDistribution(density=1e20),
        )
        sim.add_species(electrons, picmi.PseudoRandomLayout(n_macroparticles_per_cell=2))
        sim.diagnostics = diagnostics
        return sim

    @staticmethod
    def __render_binning_setup(simulation):
        with TemporaryDirectory() as outdir:
            simulation.write_input_file(outdir, exist_ok=True)

            param_path = os.path.join(outdir, "include", "picongpu", "param", "binningSetup.param")
            assert os.path.isfile(param_path), f"no rendered binningSetup.param under {outdir}"
            with open(param_path) as param_file:
                rendered = param_file.read()

            metadata_path = os.path.join(outdir, "metadata", "pypicongpu_rendering_context.json")
            assert os.path.isfile(metadata_path), f"no rendering context under {outdir}"
            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)

        return rendered, metadata

    def test_default_renders_bounded_region(self):
        sim = self.__get_simulation([self.__get_binning("default", picmi.Species(name="e", particle_type="electron"))])

        rendered, metadata = self.__render_binning_setup(sim)

        assert ".enableRegion(ParticleRegion::Bounded)" in rendered
        assert ".disableRegion(ParticleRegion::Leaving)" in rendered
        assert ".enableRegion(ParticleRegion::Leaving)" not in rendered

        binning_context = next(entry for entry in metadata["output"] if entry["type_binning"] is True)
        assert binning_context["particle_region"] == {"Bounded": True, "Leaving": False}

    def test_leaving_region_rendered(self):
        electrons = picmi.Species(
            name="e",
            particle_type="electron",
            initial_distribution=picmi.UniformDistribution(density=1e20),
        )
        binning = self.__get_binning(
            "leaving",
            electrons,
            particle_region=["Leaving"],
            # when binning leaving particles, notify starts at 1 (see binningPlugin.rst)
            period=TimeStepSpec[1:],
        )
        sim = self.__get_simulation([binning])

        rendered, metadata = self.__render_binning_setup(sim)

        assert ".enableRegion(ParticleRegion::Leaving)" in rendered
        assert ".disableRegion(ParticleRegion::Bounded)" in rendered

        binning_context = next(entry for entry in metadata["output"] if entry["type_binning"] is True)
        assert binning_context["particle_region"] == {"Bounded": False, "Leaving": True}

    def test_both_regions_rendered(self):
        electrons = picmi.Species(
            name="e",
            particle_type="electron",
            initial_distribution=picmi.UniformDistribution(density=1e20),
        )
        binning = self.__get_binning("both", electrons, particle_region={"Bounded", "Leaving"})
        sim = self.__get_simulation([binning])

        rendered, _ = self.__render_binning_setup(sim)

        assert ".enableRegion(ParticleRegion::Bounded)" in rendered
        assert ".enableRegion(ParticleRegion::Leaving)" in rendered

    def test_particle_region_passed_through_to_pypicongpu(self):
        electrons = picmi.Species(name="e", particle_type="electron")
        for particle_region, expected in [
            (None, ["Bounded"]),
            (["Leaving"], ["Leaving"]),
            ({"Leaving", "Bounded"}, ["Bounded", "Leaving"]),
        ]:
            with self.subTest(particle_region=particle_region):
                binning = self.__get_binning("roundtrip", electrons, particle_region=particle_region)
                converted = binning.get_as_pypicongpu(time_step_size=1, num_steps=16)
                assert converted.particle_region == expected

    def test_invalid_particle_region_rejected(self):
        electrons = picmi.Species(name="e", particle_type="electron")
        with self.assertRaises(ValueError):
            self.__get_binning("invalid", electrons, particle_region=["Leaving", "MiddleEarth"])
        with self.assertRaises(ValueError):
            self.__get_binning("empty", electrons, particle_region=[])
