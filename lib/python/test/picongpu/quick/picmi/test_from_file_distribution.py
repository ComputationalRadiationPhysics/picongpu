"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

import pytest
from pydantic import ValidationError
from picongpu import picmi, templates
from picongpu.picmi import FromFileDistribution
from picongpu.pypicongpu.rendering.renderer import Renderer
from picongpu.pypicongpu.species.operation.particlefromfile import ParticleFromFile

ARBITRARY_GRID = picmi.Cartesian3DGrid(
    lower_bound=[0, 0, 0],
    upper_bound=[16, 16, 16],
    number_of_cells=[16, 16, 16],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)


def get_sim(grid=ARBITRARY_GRID):
    return picmi.Simulation(
        max_steps=0,
        solver=picmi.ElectromagneticSolver(method="Yee", cfl=1.0, grid=grid),
    )


def get_species(file_path="/some/bunch.bp5", iteration=0, **kwargs):
    return picmi.Species(
        name="bunch",
        particle_type="electron",
        initial_distribution=FromFileDistribution(file_path=file_path, iteration=iteration, **kwargs),
    )


def render_species_initialization(sim) -> str:
    pypic = sim.get_as_pypicongpu()
    context = pypic.get_rendering_context()
    Renderer.check_rendering_context(context)
    preprocessed = Renderer.get_context_preprocessed(context)
    template = (
        templates.path() / "include" / "picongpu" / "param" / "speciesInitialization.param.mustache"
    ).read_text()
    return Renderer.get_rendered_template(preprocessed, template)


class TestFromFileDistributionConstruction(TestCase):
    def test_file_path_and_default_iteration(self):
        distribution = FromFileDistribution(file_path="/some/bunch.bp5")
        assert distribution.file_path == "/some/bunch.bp5"
        assert distribution.iteration == 0

    def test_iteration_is_configurable(self):
        assert FromFileDistribution(file_path="/x", iteration=7).iteration == 7

    def test_negative_iteration_rejected(self):
        with pytest.raises(ValidationError):
            FromFileDistribution(file_path="/x", iteration=-1)

    def test_is_exported(self):
        assert picmi.FromFileDistribution is FromFileDistribution


class TestFromFileDistributionRegistration(TestCase):
    def test_creates_particle_from_file_operation(self):
        sim = get_sim()
        sim.add_species(get_species(file_path="/x/bunch.bp5", iteration=2), None)
        operations = [op for op in sim.get_as_pypicongpu().init_operations if isinstance(op, ParticleFromFile)]
        assert len(operations) == 1
        assert operations[0].file_path == "/x/bunch.bp5"
        assert operations[0].iteration == 2

    def test_no_density_operation_is_registered(self):
        sim = get_sim()
        sim.add_species(get_species(), None)
        operation_types = [type(op).__name__ for op in sim.get_as_pypicongpu().init_operations]
        assert "SimpleDensity" not in operation_types
        assert "ParticleFromFile" in operation_types

    def test_layout_is_rejected_imperatively(self):
        sim = get_sim()
        with pytest.raises(ValueError, match="cannot be combined with a layout"):
            sim.add_species(get_species(), picmi.OnePositionLayout(n_macroparticles_per_cell=1))

    def test_layout_is_rejected_declaratively(self):
        with pytest.raises(ValueError, match="cannot be combined with a layout"):
            picmi.Simulation(
                max_steps=0,
                solver=picmi.ElectromagneticSolver(method="Yee", cfl=1.0, grid=ARBITRARY_GRID),
                species=[get_species()],
                layouts=[picmi.OnePositionLayout(n_macroparticles_per_cell=1)],
            )

    def test_density_scale_is_rejected(self):
        sim = get_sim()
        species = picmi.Species(
            name="bunch",
            particle_type="electron",
            initial_distribution=FromFileDistribution(file_path="/x/bunch.bp5"),
            density_scale=2.0,
        )
        with pytest.raises(ValueError, match="density_scale cannot be combined with a from-file distribution"):
            sim.add_species(species, None)

    def test_multispecies_from_file_is_rejected(self):
        sim = get_sim()
        multispecies = picmi.MultiSpecies(
            names=["electrons", "ions"],
            particle_types=["electron", "proton"],
            initial_distribution=FromFileDistribution(file_path="/x/bunch.bp5"),
        )
        with pytest.raises(ValueError, match="cannot be combined with a MultiSpecies"):
            sim.add_species(multispecies, None)

    def test_two_dimensional_grid_is_rejected(self):
        grid_2d = picmi.Cartesian2DGrid(
            lower_bound=[0, 0],
            upper_bound=[16, 16],
            number_of_cells=[16, 16],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        sim = get_sim(grid_2d)
        sim.add_species(get_species(), None)
        with pytest.raises(ValueError, match="not supported on a 2D grid"):
            sim.get_as_pypicongpu()


class TestFromFileDistributionRendering(TestCase):
    def test_renders_parameter_struct_and_functor(self):
        sim = get_sim()
        sim.add_species(get_species(file_path="/x/bunch.bp5", iteration=5), None)
        rendered = render_species_initialization(sim)
        assert "FromOpenPMDFile_species_bunch" in rendered
        assert 'static constexpr char const* filePath = "/x/bunch.bp5";' in rendered
        assert "static constexpr uint32_t iteration = 5;" in rendered
        assert "LoadFromOpenPMD<FromOpenPMDFile_species_bunch>" in rendered
        assert "CreateDensity" not in rendered
