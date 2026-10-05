"""
This file is part of PIConGPU.
Copyright 2021-2026 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

import copy
import math
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import picmistandard
import pytest
from pydantic import ValidationError
from picongpu import picmi
from picongpu import templates
from picongpu.picmi.interaction.collision import CollisionalPhysicsSetup
from picongpu.picmi.interaction.ionization.fieldionization import ADK, ADKVariant, BSI, BSIExtension, Keldysh
from picongpu.pypicongpu import customuserinput, species
from picongpu.pypicongpu.field_solver import ArbitraryOrderFDTDSolver
from picongpu.pypicongpu.rendering.renderer import Renderer


def render_min_weighting(sim) -> str:
    """Render the production particle.param template and return its MIN_WEIGHTING line."""
    pypic = sim.get_as_pypicongpu()
    context = pypic.get_rendering_context()
    Renderer.check_rendering_context(context)
    preprocessed = Renderer.get_context_preprocessed(context)
    template = (templates.path() / "include" / "picongpu" / "param" / "particle.param.mustache").read_text()
    rendered = Renderer.get_rendered_template(preprocessed, template)
    return next(line.strip() for line in rendered.splitlines() if "MIN_WEIGHTING =" in line)


def get_grid(delta_x: float, delta_y: float, delta_z: float, n: int):
    # sets delta_[x,y,z] implicitly by providing bounding box+cell count
    return picmi.Cartesian3DGrid(
        number_of_cells=[n, n, n],
        lower_bound=[0, 0, 0],
        upper_bound=list(map(lambda x: n * x, [delta_x, delta_y, delta_z])),
        # required, otherwise won't spawn
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )


def get_laser(huygens_surface_positions=None):
    # a valid, minimal GaussianLaser; optional Huygens surface override
    kwargs = {}
    if huygens_surface_positions is not None:
        kwargs["picongpu_huygens_surface_positions"] = huygens_surface_positions
    return picmi.GaussianLaser(
        wavelength=1,
        waist=2,
        duration=3,
        focal_position=[5, 4, 5],
        centroid_position=[5, -1.5, 5],
        propagation_direction=[0, 1, 0],
        polarization_direction=[0, 0, 1],
        E0=5,
        **kwargs,
    )


def get_plane_wave_laser(huygens_surface_positions=None):
    # a valid, minimal PlaneWaveLaser; optional Huygens surface override
    kwargs = {}
    if huygens_surface_positions is not None:
        kwargs["picongpu_huygens_surface_positions"] = huygens_surface_positions
    return picmi.PlaneWaveLaser(
        wavelength=1,
        duration=3,
        centroid_position=[5, -1.5, 5],
        propagation_direction=[0, 1, 0],
        polarization_direction=[0, 0, 1],
        E0=5,
        **kwargs,
    )


def get_sim_cfl_helper(
    delta_t: float | None,
    cfl: float | None,
    delta_3d: tuple[float, float, float],
    method: str,
    n: int = 100,
) -> picmi.Simulation:
    grid = get_grid(delta_3d[0], delta_3d[1], delta_3d[2], n)
    solver = picmi.ElectromagneticSolver(method=method, grid=grid, cfl=cfl)
    return picmi.Simulation(time_step_size=delta_t, solver=solver)


class TestPicmiSimulation(TestCase):
    def __get_sim(self):
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)

        return sim

    def __get_tmpdir_name(self):
        """
        get name of non-existing tmp dir which will be automatically cleaned up
        """
        name = None
        with tempfile.TemporaryDirectory() as tmpdir:
            name = tmpdir
        assert not os.path.exists(name)
        self.__to_cleanup.append(name)
        return name

    def setUp(self):
        self.sim = self.__get_sim()
        self.layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)
        self.__to_cleanup = []

        self.customData_1 = [{"test_data_1": 1}, "tag_1"]
        self.customData_2 = [{"test_data_2": 2}, "tag_2"]

    def tearDown(self):
        for dir_to_cleanup in self.__to_cleanup:
            if os.path.isdir(dir_to_cleanup):
                shutil.rmtree(dir_to_cleanup)
            assert not os.path.exists(dir_to_cleanup)

    def test_cfl_yee(self):
        # the Courant-Friedrichs-Lewy condition describes the relationship
        # between delta_t, delta_[x,y,z] and a parameter, here "cfl"
        # notably, all three can be given explicitly, though only two of the
        # three are required.
        # for practical reasons, delta_[x,y,z] has to be provided
        # this test checks the proper calculation of the cfl/delta_t

        # nothing defined if grid is empty
        sim = picmi.Simulation()
        assert sim.time_step_size is None
        sim = picmi.Simulation(time_step_size=17)
        assert sim.time_step_size == 17

        # delta_t = cfl = None -> ignored (at least during instantiation;
        # can throw later)
        get_sim_cfl_helper(None, None, (1, 1, 1), "Yee")

        # delta_t -> cfl
        sim = get_sim_cfl_helper(2.02760320328617635877e-13, None, (7e-6, 8e-6, 9e-6), "Yee")
        assert abs(sim.solver.cfl - 13.37) < 1e-10

        # cfl -> delta_t
        sim = get_sim_cfl_helper(None, 0.99, (3, 4, 5), "Yee")
        assert abs(sim.time_step_size - 7.14500557764070900528e-9) < 1e-20

        # both delta_t and cfl defined:
        # case a: silently pass if they do match
        get_sim_cfl_helper(7.14500557764070900528e-9, 0.99, (3, 4, 5), "Yee")

        # case b: raise error if no match
        with pytest.raises(ValueError):
            # delta_t does not match cfl at all
            get_sim_cfl_helper(1, 0.99, (3, 4, 5), "Yee")

    def test_huygens_surface_positions_mismatch_raises(self):
        """two lasers with differing Huygens surface positions are rejected at translate time (https://github.com/chillenzer-agents/picongpu/issues/115)"""
        sim = self.__get_sim()
        sim.add_laser(get_laser([[16, -16], [16, -16], [16, -16]]), None)
        sim.add_laser(get_laser([[1, -1], [1, -1], [1, -1]]), None)
        with pytest.raises(ValueError, match="picongpu_huygens_surface_positions"):
            sim.get_as_pypicongpu()

    def test_huygens_surface_positions_matching_ok(self):
        """identical Huygens surface positions across multiple lasers translate fine (https://github.com/chillenzer-agents/picongpu/issues/115)"""
        sim = self.__get_sim()
        sim.add_laser(get_laser([[3, -3], [3, -3], [3, -3]]), None)
        sim.add_laser(get_laser([[3, -3], [3, -3], [3, -3]]), None)
        sim.add_laser(get_laser([[3, -3], [3, -3], [3, -3]]), None)
        assert sim.get_as_pypicongpu().model_dump() != {}

    def test_huygens_surface_positions_single_laser_ok(self):
        """a single laser needs no cross-laser consistency check (https://github.com/chillenzer-agents/picongpu/issues/115)"""
        sim = self.__get_sim()
        sim.add_laser(get_laser([[9, -9], [9, -9], [9, -9]]), None)
        assert sim.get_as_pypicongpu().model_dump() != {}

    def test_huygens_surface_positions_mixed_laser_types(self):
        """the consistency check is type-agnostic and compares across laser kinds (https://github.com/chillenzer-agents/picongpu/issues/115)"""
        # differing positions across two different laser types are rejected
        sim = self.__get_sim()
        sim.add_laser(get_laser([[16, -16], [16, -16], [16, -16]]), None)
        sim.add_laser(get_plane_wave_laser([[1, -1], [1, -1], [1, -1]]), None)
        with pytest.raises(ValueError, match="picongpu_huygens_surface_positions"):
            sim.get_as_pypicongpu()

        # identical positions across the two different laser types translate fine
        sim = self.__get_sim()
        sim.add_laser(get_laser([[3, -3], [3, -3], [3, -3]]), None)
        sim.add_laser(get_plane_wave_laser([[3, -3], [3, -3], [3, -3]]), None)
        assert sim.get_as_pypicongpu().model_dump() != {}

    def test_species_translation(self):
        """test that species are moved to PyPIConGPU simulation"""
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)

        profile = picmi.UniformDistribution(density=42)
        layout3 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        layout4 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

        # species list empty by default
        assert sim.get_as_pypicongpu().species == []

        # not placed
        sim.add_species(picmi.Species(name="dummy1", mass=5), None)

        # placed with entire placement and 3ppc
        sim.add_species(picmi.Species(name="dummy2", mass=3, density_scale=4, initial_distribution=profile), layout3)

        # placed with default ratio of 1 and 4ppc
        sim.add_species(picmi.Species(name="dummy3", mass=3, initial_distribution=profile), layout4)

        picongpu = sim.get_as_pypicongpu()
        assert len(picongpu.species) == 3
        species_names = set(map(lambda species: species.name, picongpu.species))
        assert species_names == {"dummy1", "dummy2", "dummy3"}

        # check typical ppc is derived
        assert picongpu.typical_ppc == 3

    def test_declarative_species_registers_density(self):
        """constructor species/layouts must register the same density init as add_species (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)

        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        def new_species():
            return picmi.Species(name="declarative", mass=3, charge=4, initial_distribution=profile)

        declarative = picmi.Simulation(
            time_step_size=17, max_steps=4, solver=solver, species=[new_species()], layouts=[layout]
        )
        imperative = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)
        imperative.add_species(new_species(), layout)

        declarative_ops = declarative.get_as_pypicongpu().init_operations
        imperative_ops = imperative.get_as_pypicongpu().init_operations
        assert declarative_ops != []
        assert [type(op).__name__ for op in declarative_ops] == [type(op).__name__ for op in imperative_ops]

    def test_declarative_species_skips_none_distribution(self):
        """a None initial_distribution is skipped and layout-less species stay unplaced (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        placed = picmi.Species(name="placed", mass=1, initial_distribution=picmi.UniformDistribution(density=42))
        not_placed = picmi.Species(name="not_placed", mass=1)

        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=17, max_steps=4, solver=solver, species=[placed, not_placed], layouts=[layout, None]
        )

        assert len(sim.species) == 2
        assert len(sim.layouts) == 2
        assert sim.get_as_pypicongpu().init_operations != []

    def test_declarative_species_layout_without_distribution_raises(self):
        """a layout with no initial distribution is rejected as in add_species (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        with pytest.raises(Exception, match=".*initial.*distribution.*"):
            picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                species=[picmi.Species(name="dummy")],
                layouts=[layout],
            )

    def test_declarative_species_length_mismatch_raises(self):
        """species and layouts must have equal length (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        with pytest.raises(Exception, match=".*same length.*"):
            picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                species=[
                    picmi.Species(name="a", initial_distribution=picmi.UniformDistribution(density=42)),
                    picmi.Species(name="b", initial_distribution=picmi.UniformDistribution(density=42)),
                ],
                layouts=[layout],
            )

    def test_declarative_species_then_add_species_appends(self):
        """a later add_species appends to the declarative lists without double registering (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=4,
            solver=solver,
            species=[picmi.Species(name="declarative", mass=1, initial_distribution=profile)],
            layouts=[layout],
        )
        sim.add_species(picmi.Species(name="imperative", mass=1, initial_distribution=profile), layout)

        assert len(sim.species) == 2
        assert len(sim.layouts) == 2

    def test_add_species_through_plane_after_construction(self):
        """the inherited add_species_through_plane still appends after construction (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)

        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        injection = picmi.Species(
            name="injected", mass=1, charge=1, initial_distribution=picmi.UniformDistribution(density=42)
        )
        sim.add_species_through_plane(injection, layout, [0, 0, 0], [1, 0, 0])

        assert len(sim.species) == 1
        assert len(sim.layouts) == 1

    def test_declarative_species_2d(self):
        """the declarative registration also works with a 2D grid (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[64, 64],
            lower_bound=[0, 0],
            upper_bound=[64, 64],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)

        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=4,
            solver=solver,
            species=[
                picmi.Species(name="declarative2d", mass=1, initial_distribution=picmi.UniformDistribution(density=42))
            ],
            layouts=[layout],
        )

        assert sim.get_as_pypicongpu().init_operations != []

    def test_declarative_species_oneposition_layout(self):
        """OnePositionLayout is accepted by the declarative constructor too (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=4,
            solver=solver,
            species=[
                picmi.Species(name="oneposition", mass=1, initial_distribution=picmi.UniformDistribution(density=42))
            ],
            layouts=[picmi.OnePositionLayout(n_macroparticles_per_cell=2)],
        )

        assert sim.get_as_pypicongpu().init_operations != []

    def test_declarative_registration_is_stateless(self):
        """later assignments do not re-run or mutate the declarative registration (https://github.com/chillenzer-agents/picongpu/issues/189)"""
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=4,
            solver=solver,
            species=[
                picmi.Species(name="declarative", mass=1, initial_distribution=picmi.UniformDistribution(density=42))
            ],
            layouts=[layout],
        )
        registered = list(sim.species[0].get_operation_requirements())

        # Re-validating the model (any assignment) must not add or drop entries.
        sim.max_steps = 5
        assert list(sim.species[0].get_operation_requirements()) == registered

    def test_explicit_typical_ppc(self):
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_typical_ppc=15)

        profile = picmi.UniformDistribution(density=42)
        layout3 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        layout4 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

        # placed with entire placement and 3ppc
        sim.add_species(
            picmi.Species(name="dummy2", mass=3, charge=4, density_scale=4, initial_distribution=profile), layout3
        )
        # placed with default ratio of 1 and 4ppc
        sim.add_species(picmi.Species(name="dummy3", mass=3, charge=4, initial_distribution=profile), layout4)

        picongpu = sim.get_as_pypicongpu()
        assert len(picongpu.species) == 2
        species_names = set(map(lambda species: species.name, picongpu.species))
        assert species_names == {"dummy2", "dummy3"}

        # check explicitly set typical ppc is respected
        assert picongpu.typical_ppc == 15

    def test_wrong_explicitly_set_typical_ppc(self):
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        wrongValues = [0, -1, -15]
        for value in wrongValues:
            with pytest.raises(ValueError, match="Typical ppc should be > 0"):
                picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_typical_ppc=value)

        wrongTypes = [0.0, -1.0, -15.0]
        for value in wrongTypes:
            with pytest.raises(ValueError, match="Typical ppc should be > 0"):
                picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_typical_ppc=value)

    def test_min_weighting_default_renders_as_float(self):
        """unset picongpu_min_weighting falls back to the C++ default 10.0 (float literal)"""
        assert render_min_weighting(self.sim) == "constexpr float_X MIN_WEIGHTING = 10.0;"

    def test_min_weighting_explicit_renders_as_float(self):
        """an explicit picongpu_min_weighting is threaded into the rendered MIN_WEIGHTING"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_min_weighting=2.0)
        assert render_min_weighting(sim) == "constexpr float_X MIN_WEIGHTING = 2.0;"

    def test_min_weighting_rejects_non_positive_and_non_finite(self):
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        for value in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with pytest.raises(ValidationError, match="Minimum weighting must be finite and > 0"):
                picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_min_weighting=value)

    def test_pypicongpu_min_weighting_rejects_non_positive_and_non_finite(self):
        """the pypicongpu model validates directly, not only via the PICMI surface"""
        pypic = self.sim.get_as_pypicongpu()
        for value in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with pytest.raises(ValidationError, match="Minimum weighting must be finite and > 0"):
                type(pypic)(**{**pypic.__dict__, "min_weighting": value})

    def test_invalid_placement(self):
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)

        # both profile and layout must be given
        with pytest.raises(Exception, match=".*initial.*distribution.*"):
            # no profile
            sim = copy.deepcopy(self.sim)
            sim.add_species(picmi.Species(name="dummy3"), layout)
            sim.get_as_pypicongpu()
        with pytest.raises(Exception, match=".*layout.*"):
            # no layout
            sim = copy.deepcopy(self.sim)
            sim.add_species(picmi.Species(name="dummy3", initial_distribution=profile), None)
            sim.get_as_pypicongpu()

        with pytest.raises(Exception, match=".*initial.*distribution.*"):
            # neither profile nor layout, but ratio
            sim = copy.deepcopy(self.sim)
            sim.add_species(picmi.Species(name="dummy3", density_scale=7), None)
            sim.get_as_pypicongpu()

    def test_operations_simple_density_translated(self):
        """simple density operations are correctly derived

        Species are initialised independently by default (picmi-standard
        semantics, Option B): only members of the same explicit MultiSpecies
        share a density operation (collective initialisation), everything else
        is split into individual operations.
        """
        profile = picmi.UniformDistribution(density=42)
        other_profile = picmi.UniformDistribution(density=17)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        other_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

        # explicitly coordinated group, added as a whole (standard PICMI form)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H"],
            names=["colocated1", "colocated2"],
            proportions=[4, 2],
            initial_distribution=profile,
        )
        self.sim.add_species(species=multispecies, layout=layout)
        # independent species that merely look the same w.r.t. profile/layout
        self.sim.add_species(
            picmi.Species(name="separate1", mass=3, initial_distribution=other_profile),
            layout,
        )
        self.sim.add_species(
            picmi.Species(name="separate2", mass=4, initial_distribution=profile),
            other_layout,
        )

        pypic = self.sim.get_as_pypicongpu()
        my_species = pypic.species
        operations = pypic.init_operations

        # species
        assert len(my_species) == 4
        assert ["colocated1", "colocated2", "separate1", "separate2"] == list(
            map(lambda species: species.name, my_species)
        )

        # operations
        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                operations,
            )
        )
        assert len(density_operations) == 3
        for op in density_operations:
            assert isinstance(op.profile, species.operation.densityprofile.Uniform)

            species_names = set(map(lambda species: species.name, op.species))

            # ensure grouping:
            if "separate1" in species_names or "separate2" in species_names:
                # one of the two lone species (no MultiSpecies -> independent)
                assert len(species_names) == 1
            else:
                # the two colocated species (explicit MultiSpecies)
                assert len(species_names) == 2

            # check profile
            if "separate2" in species_names or "colocated1" in species_names:
                # used "profile"
                assert op.profile.density_si == 42
            else:
                # used "other_profile"
                assert op.profile.density_si == 17

            # check layout
            if "separate1" in species_names or "colocated1" in species_names:
                # used "layout"
                assert op.layout.ppc == 3
            else:
                # used "other_layout"
                assert op.layout.ppc == 4

    def test_simple_density_independent_by_default(self):
        """same distribution+layout do not merge: standalone species are independent (Option B)"""
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        for name in ("electron_1", "electron_2"):
            self.sim.add_species(
                picmi.Species(name=name, particle_type="electron", initial_distribution=profile), layout
            )

        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                self.sim.get_as_pypicongpu().init_operations,
            )
        )
        assert len(density_operations) == 2

    def test_multispecies_three_members_one_created_two_derived(self):
        """#5762 shape: 3 members, same density but differing momentum -> 1 created + 2 derived"""
        base_profile = picmi.UniformDistribution(density=42)
        momenta = [
            picmi.UniformDistribution(density=42, rms_velocity=[0.0, 0.0, 0.0]),
            picmi.UniformDistribution(density=42, rms_velocity=[1.0e6, 1.0e6, 1.0e6]),
            picmi.UniformDistribution(density=42, rms_velocity=[2.0e6, 1.0e6, 5.0e5]),
        ]
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H", "He"],
            names=["a", "b", "c"],
            proportions=[1.0, 1.0, 1.0],
            initial_distribution=base_profile,
        )
        for member, distribution in zip(multispecies, momenta, strict=True):
            member.initial_distribution = distribution
        # add the whole group at once with one shared layout
        self.sim.add_species(species=multispecies, layout=layout)

        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                self.sim.get_as_pypicongpu().init_operations,
            )
        )
        # exactly ONE CreateDensity placing the *single* created species...
        assert len(density_operations) == 1
        op = density_operations[0]
        assert isinstance(op.placed_species_initial, species.Species)
        assert len(op.placed_species_copied) == 2
        # ... plus per-species momentum (SimpleMomentum) for all three members
        momentum_operations = list(
            filter(
                lambda op_: isinstance(op_, species.operation.SimpleMomentum),
                self.sim.get_as_pypicongpu().init_operations,
            )
        )
        assert len(momentum_operations) == 3

    def test_multispecies_grouping_is_structural(self):
        """grouping is structural: the Simulation stores the MultiSpecies entry as-is"""
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H"],
            names=["electron", "proton"],
            proportions=[1.0, 1.0],
            initial_distribution=profile,
        )
        sim = picmi.Simulation(
            time_step_size=17, max_steps=4, solver=self.sim.solver, species=[multispecies], layouts=[layout]
        )
        # one entry, stored as given -- not expanded into its members
        assert sim.species == [multispecies]
        assert sim.layouts == [layout]
        assert len(multispecies) == 2

    def test_multispecies_members_and_density_ratio(self):
        """MultiSpecies members share one distribution and map proportions to DensityRatio"""
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=8)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H"],
            names=["electron", "proton"],
            proportions=[1.0, 2.0],
            initial_distribution=profile,
        )
        assert len(multispecies) == 2

        for member, proportion in zip(multispecies, [1.0, 2.0], strict=True):
            # plain picmi species sharing the distribution
            assert isinstance(member, picmi.Species)
            assert member.initial_distribution is profile
            assert member.density_scale == proportion
            # density_scale maps to DensityRatio on the pypicongpu level
            assert member._evaluate_species_requirements()["constants"] and any(
                ratio.ratio == proportion
                for ratio in member._evaluate_species_requirements()["constants"]
                if hasattr(ratio, "ratio")
            )

        # the whole MultiSpecies is added as one object with a single shared layout
        self.sim.add_species(species=multispecies, layout=layout)

        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                self.sim.get_as_pypicongpu().init_operations,
            )
        )
        assert len(density_operations) == 1
        op = density_operations[0]
        # exactly one species is created ("placed"), the other one is derived
        assert len(op.placed_species_copied) == 1
        assert set(s.name for s in op.species) == {"electron", "proton"}

    def test_multispecies_added_as_whole_object(self):
        """the standard interface adds the MultiSpecies in one add_species call

        Mirrors the PICMI-standard example (``sim.add_species(species=multi,
        layout=layout)``): the object is stored as one entry carrying the shared
        layout and is collectively initialised as a whole.
        """
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)
        multispecies = picmi.MultiSpecies(
            particle_types=["H", "electron"],
            names=["ions", "electrons"],
            proportions=[1.0, 1.0],
            initial_distribution=profile,
        )

        self.sim.add_species(species=multispecies, layout=layout)

        # the whole group is one entry with the one shared layout
        assert self.sim.species == [multispecies]
        assert self.sim.layouts == [layout]

        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                self.sim.get_as_pypicongpu().init_operations,
            )
        )
        # collective initialisation: a single density operation for both members
        assert len(density_operations) == 1
        assert set(s.name for s in density_operations[0].species) == {"ions", "electrons"}

    def test_multispecies_added_as_whole_object_declaratively(self):
        """the declarative constructor accepts a MultiSpecies as one species

        ``Simulation(species=[multi], layouts=[layout])``: the group is stored as
        one entry carrying the shared layout and is collectively initialised --
        the declarative equivalent of the ``add_species`` form.
        """
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)
        multispecies = picmi.MultiSpecies(
            particle_types=["H", "electron"],
            names=["ions", "electrons"],
            proportions=[1.0, 1.0],
            initial_distribution=profile,
        )

        sim = picmi.Simulation(
            time_step_size=17, max_steps=4, solver=self.sim.solver, species=[multispecies], layouts=[layout]
        )

        assert sim.species == [multispecies]
        assert sim.layouts == [layout]

        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                sim.get_as_pypicongpu().init_operations,
            )
        )
        assert len(density_operations) == 1
        assert set(s.name for s in density_operations[0].species) == {"ions", "electrons"}

    def test_multispecies_members_addressed_by_name_and_index(self):
        """members are addressable by name or index, as in the standard example"""
        profile = picmi.UniformDistribution(density=42)
        multispecies = picmi.MultiSpecies(
            particle_types=["He", "Ar", "electron"],
            names=["He+", "Argon", "e-"],
            charge_states=[1, 5, None],
            proportions=[0.2, 0.8, 0.2 + 5 * 0.8],
            initial_distribution=profile,
        )
        assert multispecies["Argon"] is multispecies[1]
        assert multispecies["e-"] is multispecies[-1]

    def test_multispecies_same_density_different_momentum_charge_neutral(self):
        """#5762: same density, differing momentum -> grouped (charge-neutral)"""
        # density distribution common to all members, different momenta per member
        base_profile = picmi.UniformDistribution(density=42)
        profile_cold = picmi.UniformDistribution(density=42, rms_velocity=[0.0, 0.0, 0.0])
        profile_hot = picmi.UniformDistribution(density=42, rms_velocity=[1.0e6, 1.0e6, 1.0e6])
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H"],
            names=["colder", "hotter"],
            proportions=[1.0, 1.0],
            initial_distribution=base_profile,
        )
        # customize the momentum (rms_velocity) per member while keeping the density
        multispecies[0].initial_distribution = profile_cold
        multispecies[1].initial_distribution = profile_hot
        self.sim.add_species(species=multispecies, layout=layout)

        pypic = self.sim.get_as_pypicongpu()
        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                pypic.init_operations,
            )
        )
        momentum_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleMomentum),
                pypic.init_operations,
            )
        )
        # differing momentum must NOT prevent collective (charge-neutral) init:
        # one density op placing both species on identical in-cell positions...
        assert len(density_operations) == 1
        assert len(density_operations[0].species) == 2
        # ... while momentum is still applied individually per species
        assert len(momentum_operations) == 2

    def test_members_added_separately_are_independent(self):
        """adding MultiSpecies members individually makes them independent entries"""
        profile = picmi.UniformDistribution(density=42)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H"],
            names=["electron", "proton"],
            proportions=[1.0, 1.0],
            initial_distribution=profile,
        )
        # Adding the members one by one is *not* collective: each becomes its own
        # entry and thereby its own density operation (grouping is structural).
        for member in multispecies:
            self.sim.add_species(member, picmi.PseudoRandomLayout(n_macroparticles_per_cell=4))

        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                self.sim.get_as_pypicongpu().init_operations,
            )
        )
        assert len(density_operations) == 2
        for op in density_operations:
            assert len(op.species) == 1

    def test_multispecies_grouping_survives_deepcopy(self):
        """explicit MultiSpecies grouping survives a simulation round-trip (deepcopy)"""
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)
        multispecies = picmi.MultiSpecies(
            particle_types=["electron", "H"],
            names=["electron", "proton"],
            proportions=[1.0, 1.0],
            initial_distribution=profile,
        )
        self.sim.add_species(species=multispecies, layout=layout)

        sim_copy = copy.deepcopy(self.sim)
        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                sim_copy.get_as_pypicongpu().init_operations,
            )
        )
        # grouping must survive the copy, i.e. the two members still share one op
        assert len(density_operations) == 1
        assert len(density_operations[0].species) == 2

    def test_operation_not_placed_translated(self):
        """non-placed species are correctly translated"""
        self.sim.add_species(picmi.Species(name="notplaced", mass=1, initial_distribution=None), None)

        pypicongpu = self.sim.get_as_pypicongpu()

        assert len(pypicongpu.species) == 1
        # not placed, momentum (both initialize to empty)
        assert len(pypicongpu.init_operations) == 0

    def test_operation_momentum(self):
        """operation for momentum correctly derived from species"""
        self.sim.add_species(
            picmi.Species(
                name="valid",
                mass=17,
                initial_distribution=picmi.UniformDistribution(
                    density=17,
                    rms_velocity=[17, 17, 17],
                    directed_velocity=[31283745.0, 45132121.0, 215484563.0],
                ),
            ),
            picmi.PseudoRandomLayout(n_macroparticles_per_cell=2),
        )

        pypicongpu = self.sim.get_as_pypicongpu()

        mom_ops = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleMomentum),
                pypicongpu.init_operations,
            )
        )

        # momentum operation must always be generated
        assert len(mom_ops) == 1
        mom_op = mom_ops[0]

        assert mom_op.species.name == "valid"
        assert abs(mom_op.temperature.temperature_kev - 3.06645343e19) < 1e13
        assert mom_op.drift.direction_normalized == (
            0.14068221552237223,
            0.2029580145696681,
            0.9690286675623457,
        )
        assert abs(mom_op.drift.gamma - 1.491037242289643) < 1e-10

    def test_moving_window(self):
        """test that the user may set moving window"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            lower_bound=[0, 0, 0],
            upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=1.39e-16, max_steps=int(2048), solver=solver, picongpu_moving_window_move_point=0.9
        )
        pypic = sim.get_as_pypicongpu()

        assert abs(pypic.moving_window.move_point - 0.9) < 1e-10
        assert pypic.moving_window.stop_iteration is None

    def test_add_ionization_model(self):
        """ionization model is added correctly"""
        e = picmi.Species(name="e", particle_type="electron")
        ion1 = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
        ion2 = picmi.Species(name="nitrogen", particle_type="N", charge_state=+2)

        ionization_model_1 = ADK(
            ADK_variant=ADKVariant.LinearPolarization,
            ionization_current=None,
            ion_species=ion1,
            ionization_electron_species=e,
        )
        ionization_model_2 = ADK(
            ADK_variant=ADKVariant.LinearPolarization,
            ionization_current=None,
            ion_species=ion2,
            ionization_electron_species=e,
        )
        interaction = [ionization_model_1, ionization_model_2]

        sim = self.sim
        sim.add_species(e, None)
        sim.add_species(ion1, None)
        sim.add_species(ion2, None)

        # in use should be set via simulation constructor
        sim.interactions = interaction

        pypic_sim = sim.get_as_pypicongpu()
        operations = pypic_sim.init_operations

        # Every SetChargeState op must carry the charge state that was requested for its
        # species, and every species with a requested charge state must be matched by
        # exactly one op. Name-keyed lookup is deliberately avoided here: a rename or a
        # value mismatch must fail the assertions instead of silently bypassing them.
        set_charge_state_ops = [op for op in operations if isinstance(op, species.operation.SetChargeState)]
        expected_charge_states = {ion.name: ion.charge_state for ion in (ion1, ion2)}
        assert len(set_charge_state_ops) == len(expected_charge_states)
        for op in set_charge_state_ops:
            assert op.species.name in expected_charge_states, f"no charge state requested for {op.species.name=}"
            assert op.charge_state == expected_charge_states.pop(op.species.name)
        assert not expected_charge_states, f"missing SetChargeState op for {expected_charge_states}"

    def test_write_input_file(self):
        """sanity check picmi upstream: write input file"""
        sim = self.sim
        outdir = self.__get_tmpdir_name()
        assert not os.path.isdir(outdir)
        sim.write_input_file(outdir)
        assert os.path.isdir(outdir)
        assert os.path.exists(outdir + "/include/picongpu/param/simulation.param")

    def test_write_input_file_regenerates_existing_setup(self):
        """regenerating input into an already generated setup dir overwrites the old files (#5752)"""
        sim = self.sim
        outdir = self.__get_tmpdir_name()
        sim.write_input_file(outdir)

        # a rendered file from a nested template dir, rendered by the runner itself
        nested_file = outdir + "/etc/picongpu/N.cfg"
        assert os.path.exists(nested_file)
        with open(nested_file) as file:
            nested_content = file.read()
        # a param rendered from a default template
        param_file = outdir + "/include/picongpu/param/simulation.param"
        assert os.path.exists(param_file)
        with open(param_file) as file:
            param_content = file.read()

        # simulate stale/modified output from the previous generation
        for file in (nested_file, param_file):
            with open(file, "w") as f:
                f.write("stale")

        sim.write_input_file(outdir, exist_ok=True)

        with open(nested_file) as file:
            assert file.read() == nested_content
        with open(param_file) as file:
            assert file.read() == param_content

    def test_custom_template_dir_basic_write_input_file(self):
        """providing custom template dir possible or write_input_file"""
        # note: automatically cleaned up in teardown
        out_dir = self.__get_tmpdir_name()

        with tempfile.TemporaryDirectory() as tmpdir:
            # create test template dir
            # -> use include/picongpu,
            #    because pic-create does not copy every dir
            os.makedirs(tmpdir + "/include/picongpu")
            with open(tmpdir + "/include/picongpu/time_steps.mustache", "w") as testfile:
                testfile.write("{{{time_steps}}}")

            grid = get_grid(1, 1, 1, 32)
            solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
            # explicitly set to None
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=128,
                solver=solver,
                picongpu_template_dir=tmpdir,
            )
            sim.write_input_file(out_dir)

        # check for generated (rendered) dir
        assert os.path.isfile(out_dir + "/include/picongpu/time_steps")
        with open(out_dir + "/include/picongpu/time_steps") as rendered_file:
            assert rendered_file.read() == "128"

        # JSON has been dumped
        assert os.path.isfile(out_dir + "/metadata/pypicongpu_rendering_context.json")
        assert os.path.isfile(out_dir + "/metadata/pypicongpu_runner.json")
        assert os.path.isfile(out_dir + "/metadata/rc_params.json")

    def test_custom_input_basic_write_input_file(self):
        """test custom input may be rendered"""
        # note: automatically cleaned up in teardown
        out_dir = self.__get_tmpdir_name()

        # create bare bone PICMI-simulation
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=128,
            solver=solver,
        )

        # add custom input
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput({"test_data_1": 1}, "tag_1")
        i_2.addToCustomInput({"test_data_2": 2}, "tag_2")

        sim.picongpu_add_custom_user_input(i_1)
        sim.picongpu_add_custom_user_input(i_2)

        # write simulation
        sim.write_input_file(out_dir)

        # check for generated (rendered) dir
        assert os.path.isdir(out_dir)

        # JSON has been dumped
        assert os.path.isfile(out_dir + "/metadata/pypicongpu_rendering_context.json")
        assert os.path.isfile(out_dir + "/metadata/pypicongpu_runner.json")
        assert os.path.isfile(out_dir + "/metadata/rc_params.json")

    def test_custom_template_dir_basic_get_runner(self):
        """using picongpu_get_runner() directly sets template dir"""
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = get_grid(1, 1, 1, 32)
            solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
            # explicitly set to None
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=128,
                solver=solver,
                picongpu_template_dir=tmpdir,
            )
            runner = sim.picongpu_get_runner()

            assert list(map(Path.absolute, runner.template_dir)) == [Path(tmpdir).absolute()]

    def test_custom_template_dir_optional(self):
        """custom template dir is optional"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        # explicitly set to None
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_template_dir=None)

        # simulation is valid
        assert self.sim.get_as_pypicongpu().get_rendering_context() != {}
        runner = sim.picongpu_get_runner()

        # good default template dir is selected
        assert runner.template_dir is not None
        assert runner.template_dir != ""

    def test_custom_template_dir_checks(self):
        """sanity checks are run on template dir"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        # existing dir is ok:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir_name = tmpdir
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                picongpu_template_dir=template_dir_name,
            )

            assert sim.get_as_pypicongpu().get_rendering_context() != {}
            # no throw:
            sim.picongpu_get_runner()

        # left "with" block -- tmpdir is now deleted
        # -> now raises
        with pytest.raises(Exception, match=".*template.*"):
            picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                picongpu_template_dir=template_dir_name,
            )

    def test_custom_template_dir_types(self):
        """custom template dir is typechecked"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        valid_paths = [None, "/", Path("/")]
        for valid_path in valid_paths:
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                picongpu_template_dir=valid_path,
            )
            assert sim.get_as_pypicongpu().get_rendering_context() != {}
            # no throw:
            sim.picongpu_get_runner()

        invalid_paths = [1, 42.0]
        for invalid_path in invalid_paths:
            with pytest.raises((ValidationError, ValueError)):
                picmi.Simulation(
                    time_step_size=17,
                    max_steps=4,
                    solver=solver,
                    picongpu_template_dir=invalid_path,
                )

    def test_custom_input_pass_thru(self):
        i = customuserinput.CustomUserInput()

        i.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i.addToCustomInput(self.customData_2[0], self.customData_2[1])

        self.sim.picongpu_add_custom_user_input(i)

        renderingContextGoodResult = {"test_data_1": 1, "test_data_2": 2, "tags": ["tag_1", "tag_2"]}
        assert renderingContextGoodResult == self.sim.get_as_pypicongpu().get_rendering_context()["customuserinput"]

    def test_combination_of_several_custom_inputs(self):
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_2.addToCustomInput(self.customData_2[0], self.customData_2[1])

        self.sim.picongpu_add_custom_user_input(i_1)
        self.sim.picongpu_add_custom_user_input(i_2)

        renderingContextGoodResult = {"test_data_1": 1, "test_data_2": 2, "tags": ["tag_1", "tag_2"]}
        assert renderingContextGoodResult == self.sim.get_as_pypicongpu().get_rendering_context()["customuserinput"]

    def test_duplicated_tag_over_different_custom_inputs(self):
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_2.addToCustomInput(self.customData_2[0], self.customData_1[1])

        self.sim.picongpu_add_custom_user_input(i_1)
        self.sim.picongpu_add_custom_user_input(i_2)

        with pytest.raises(ValueError):
            self.sim.get_as_pypicongpu().get_rendering_context()

    def test_duplicated_key_over_different_custom_inputs(self):
        i = customuserinput.CustomUserInput()
        i_sameValue = customuserinput.CustomUserInput()
        i_differentValue = customuserinput.CustomUserInput()

        duplicateKeyData_differentValue = {"test_data_1": 3}
        duplicateKeyData_sameValue = {"test_data_1": 1}

        i.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_sameValue.addToCustomInput(duplicateKeyData_sameValue, "tag_2")
        i_differentValue.addToCustomInput(duplicateKeyData_differentValue, "tag_3")

        self.sim.picongpu_add_custom_user_input(i)

        # should work
        self.sim.picongpu_add_custom_user_input(i_sameValue)
        self.sim.get_as_pypicongpu().get_rendering_context()

        with pytest.raises(ValueError, match="Key test_data_1 exist already, and specified values differ."):
            self.sim.picongpu_add_custom_user_input(i_differentValue)
            self.sim.get_as_pypicongpu().get_rendering_context()

    def test_maxwell_solver_method_acceptance(self):
        """the standard solvers and the PIConGPU 'other:' extensions are accepted"""
        grid = get_grid(1, 1, 1, 32)
        for method in ("Yee", "Lehe", "CKC", "other:None"):
            assert picmi.ElectromagneticSolver(method=method, grid=grid).method == method
        # the arbitrary-order FDTD requires a stencil order
        assert (
            picmi.ElectromagneticSolver(method="other:ArbitraryOrderFDTD", grid=grid, stencil_order=[4, 4, 4]).method
            == "other:ArbitraryOrderFDTD"
        )

    def test_maxwell_solver_method_rejection(self):
        """standard methods PIConGPU does not implement are rejected"""
        grid = get_grid(1, 1, 1, 32)
        for method in ("PSTD", "PSATD", "GPSTD", "DS", "ECT", "other:Substepping", "Foo"):
            with pytest.raises(ValidationError):
                picmi.ElectromagneticSolver(method=method, grid=grid)

    def test_maxwell_solver_stencil_order_uniformity(self):
        """stencil_order must be all-axes-equal and maps to neighbors = order // 2"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="other:ArbitraryOrderFDTD", grid=grid, stencil_order=[4, 4, 4])
        # order 4 -> 2 neighbors
        assert isinstance(solver.get_as_pypicongpu(), ArbitraryOrderFDTDSolver)
        assert solver.get_as_pypicongpu().name == "ArbitraryOrderFDTD<2>"
        # order 6 -> 3 neighbors
        solver = picmi.ElectromagneticSolver(method="other:ArbitraryOrderFDTD", grid=grid, stencil_order=[6, 6, 6])
        assert solver.get_as_pypicongpu().name == "ArbitraryOrderFDTD<3>"
        # non-uniform, sub-2, odd, and empty orders are all rejected
        for bad in ([2, 4, 4], [4, 4, 6], [1, 1, 1], [5, 5, 5], [3, 3, 3], []):
            with pytest.raises(ValidationError):
                picmi.ElectromagneticSolver(method="other:ArbitraryOrderFDTD", grid=grid, stencil_order=bad)

    def test_maxwell_solver_stencil_order_fixed_order_rejected(self):
        """a stencil_order is meaningless on a fixed-order solver and is rejected"""
        grid = get_grid(1, 1, 1, 32)
        for method in ("Yee", "Lehe", "CKC", "other:None"):
            with pytest.raises(ValidationError):
                picmi.ElectromagneticSolver(method=method, grid=grid, stencil_order=[4, 4, 4])
        # and the arbitrary-order FDTD needs one
        with pytest.raises(ValidationError):
            picmi.ElectromagneticSolver(method="other:ArbitraryOrderFDTD", grid=grid)

    def test_none_solver_stays_out_of_cfl_gate(self):
        """the None solver has no CFL limit: cfl/delta_t are left untouched"""
        grid = get_grid(1, 2, 3, 10)
        # nothing given -> both stay None
        sim = picmi.Simulation(solver=picmi.ElectromagneticSolver(method="other:None", grid=grid))
        assert sim.time_step_size is None
        assert sim.solver.cfl is None
        # any delta_t is legal and cfl stays None (not derived)
        sim = picmi.Simulation(time_step_size=1.0, solver=picmi.ElectromagneticSolver(method="other:None", grid=grid))
        assert sim.time_step_size == 1.0
        assert sim.solver.cfl is None
        # a given cfl is never cross-checked / derived against delta_t
        sim = picmi.Simulation(
            time_step_size=1.0, solver=picmi.ElectromagneticSolver(method="other:None", grid=grid, cfl=0.99)
        )
        assert sim.time_step_size == 1.0
        assert sim.solver.cfl == 0.99

    def test_maxwell_solver_pypicongpu_translation(self):
        """each method maps to the right pypicongpu field-solver model and name"""
        grid = get_grid(1, 1, 1, 32)
        expected = {
            "Yee": "Yee",
            "Lehe": "Lehe<>",
            "CKC": "CKC",
            "other:None": "None",
        }
        for method, name in expected.items():
            assert picmi.ElectromagneticSolver(method=method, grid=grid).get_as_pypicongpu().name == name

    def test_maxwell_solver_cfl_per_solver(self):
        """CKC uses the minimum cell and the AO FDTD scales the Yee limit by its weight sum"""
        import math

        from picongpu.picmi import constants

        delta_3d = (3.0, 4.0, 5.0)
        n = 100
        min_cell = min(delta_3d)
        yee_cfl_limit = 1 / math.sqrt(1 / 3.0**2 + 1 / 4.0**2 + 1 / 5.0**2)
        ao_order4_limit = yee_cfl_limit / picmi.solver._ao_fDTD_weight_sum(2)  # neighbors = 4 // 2 = 2

        # CKC: cfl = c * dt / min_cell  ->  dt = cfl * min_cell / c
        sim = picmi.Simulation(solver=picmi.ElectromagneticSolver(method="CKC", grid=get_grid(*delta_3d, n=n), cfl=0.9))
        assert abs(sim.time_step_size - 0.9 * min_cell / constants.c) < 1e-30

        # AO FDTD order 4: cfl = c * dt / ao_limit  ->  dt = cfl * ao_limit / c
        sim = picmi.Simulation(
            solver=picmi.ElectromagneticSolver(
                method="other:ArbitraryOrderFDTD", grid=get_grid(*delta_3d, n=n), stencil_order=[4, 4, 4], cfl=0.99
            )
        )
        assert math.isclose(sim.time_step_size, 0.99 * ao_order4_limit / constants.c, rel_tol=1e-12, abs_tol=0.0)

        # CKC mismatch still raises
        good = picmi.Simulation(
            solver=picmi.ElectromagneticSolver(method="CKC", grid=get_grid(*delta_3d, n=n), cfl=0.9)
        )
        with pytest.raises(ValueError):
            picmi.Simulation(
                time_step_size=1.0,
                solver=picmi.ElectromagneticSolver(method="CKC", grid=get_grid(*delta_3d, n=n), cfl=0.9),
            )
        # and the matching value passes
        picmi.Simulation(
            time_step_size=good.time_step_size,
            solver=picmi.ElectromagneticSolver(method="CKC", grid=get_grid(*delta_3d, n=n), cfl=0.9),
        )


def _interaction_sim():
    """Build a minimal simulation with one ion and its electron product species."""
    sim = picmi.Simulation(
        time_step_size=17,
        max_steps=4,
        solver=picmi.ElectromagneticSolver(method="Yee", grid=get_grid(1, 1, 1, 32)),
    )
    e = picmi.Species(name="e", particle_type="electron")
    ion = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
    sim.add_species(e, None)
    sim.add_species(ion, None)
    return sim, ion, e


def _render_species_definition(sim) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "input")
        sim.write_input_file(output_dir)
        rendered_path = os.path.join(output_dir, "include", "picongpu", "param", "speciesDefinition.param")
        with open(rendered_path) as rendered_file:
            return rendered_file.read()


class TestAddInteraction:
    def test_keldysh_bare(self):
        """a bare standard FieldIonization with model=Keldysh renders the Keldysh model"""
        sim, ion, e = _interaction_sim()
        sim.add_interaction(picmi.FieldIonization(model="Keldysh", ionized_species=ion, product_species=e))

        model = sim.interactions[0]
        assert isinstance(model, Keldysh)
        # the standard names are mapped onto the concrete model
        assert model.ion_species is ion
        assert model.ionization_electron_species is e
        # a bare standard field ionization defaults to the C++ current::None
        assert model.ionization_current is None

        rendered = _render_species_definition(sim)
        assert "Keldysh" in rendered
        assert "particles::ionization::current::None" in rendered

    def test_adk_with_variant(self):
        """the ADK model carries the supplied ADK variant"""
        sim, ion, e = _interaction_sim()
        sim.add_interaction(
            picmi.FieldIonization(
                model="ADK", ionized_species=ion, product_species=e, ADK_variant=ADKVariant.LinearPolarization
            )
        )
        model = sim.interactions[0]
        assert isinstance(model, ADK)
        assert model.ADK_variant is ADKVariant.LinearPolarization

    def test_bsi_with_extensions(self):
        """the BSI model carries the supplied BSI extensions"""
        sim, ion, e = _interaction_sim()
        sim.add_interaction(
            picmi.FieldIonization(
                model="BSI", ionized_species=ion, product_species=e, BSI_extensions=[BSIExtension.StarkShift]
            )
        )
        model = sim.interactions[0]
        assert isinstance(model, BSI)
        assert model.BSI_extensions == (BSIExtension.StarkShift,)

    def test_adk_missing_variant_raises(self):
        """the ADK model requires an ADK variant and raises a clear error without one"""
        sim, ion, e = _interaction_sim()
        with pytest.raises(ValueError, match="ADK_variant"):
            sim.add_interaction(picmi.FieldIonization(model="ADK", ionized_species=ion, product_species=e))

    def test_bsi_missing_extensions_raises(self):
        """the BSI model requires extensions and raises a clear error without them"""
        sim, ion, e = _interaction_sim()
        with pytest.raises(ValueError, match="BSI_extensions"):
            sim.add_interaction(picmi.FieldIonization(model="BSI", ionized_species=ion, product_species=e))

    @pytest.mark.parametrize(
        "model_name",
        ["ADK", "adk", "Adk", "BSI", "bsi", "Keldysh", "keldysh", "KeLDySh"],
    )
    def test_model_name_case_insensitive(self, model_name):
        """model selection matches the MODEL_NAME constants case-insensitively"""
        _, ion, e = _interaction_sim()
        field_ionization = picmi.FieldIonization(
            model=model_name,
            ionized_species=ion,
            product_species=e,
            ADK_variant=ADKVariant.LinearPolarization,
            BSI_extensions=[BSIExtension.StarkShift],
        )
        expected = {"adk": ADK, "bsi": BSI, "keldysh": Keldysh}[model_name.lower()]
        assert field_ionization._resolve_model_class() is expected

    def test_concrete_models_are_picmi_interactions(self):
        """the concrete ionization models are accepted by the standard interactions field"""
        from picmistandard import PICMI_Interaction

        for model_class in (ADK, BSI, Keldysh):
            assert issubclass(model_class, PICMI_Interaction)
        assert issubclass(picmi.FieldIonization, picmistandard.PICMI_FieldIonization)

    def test_constructor_interactions_accepts_field_ionization(self):
        """the standard interactions=[...] constructor parameter is a first-class entry point"""
        e = picmi.Species(name="e", particle_type="electron")
        ion = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=4,
            solver=picmi.ElectromagneticSolver(method="Yee", grid=get_grid(1, 1, 1, 32)),
            species=[ion, e],
            layouts=[None, None],
            interactions=[picmi.FieldIonization(model="Keldysh", ionized_species=ion, product_species=e)],
        )
        assert isinstance(sim.interactions[0], Keldysh)
        assert "Keldysh" in _render_species_definition(sim)

    def test_constructor_interactions_accepts_plain_standard(self):
        """a plain picmistandard.PICMI_FieldIonization in the constructor list is mapped too"""
        e = picmi.Species(name="e", particle_type="electron")
        ion = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=4,
            solver=picmi.ElectromagneticSolver(method="Yee", grid=get_grid(1, 1, 1, 32)),
            species=[ion, e],
            layouts=[None, None],
            interactions=[picmistandard.PICMI_FieldIonization(model="Keldysh", ionized_species=ion, product_species=e)],
        )
        assert isinstance(sim.interactions[0], Keldysh)

    def test_add_interaction_accepts_collisions_and_synchrotron(self):
        """add_interaction accepts the same types as the constructor list, not only field ionization"""
        sim, ion, e = _interaction_sim()
        photon = picmi.Species(name="photons", particle_type="photon")
        sim.add_species(photon, None)

        synchrotron = picmi.Synchrotron(electron_species=e, photon_species=photon)
        sim.add_interaction(synchrotron)
        assert sim.interactions == [synchrotron]

        collision = picmi.Collision.construct_all_to_all([e, ion], functor=picmi.ConstLogCollision(coulomb_log=2.0))
        sim.add_interaction(collision)
        # the bare collision is merged into a CollisionalPhysicsSetup by the shared pipeline
        assert isinstance(sim.interactions[-1], CollisionalPhysicsSetup)
        assert sim.interactions[-1].collisions == [collision]

    def test_unsupported_standard_interaction_raises(self):
        """a standard interaction type PIConGPU does not support is rejected, not silently dropped"""
        from picmistandard import PICMI_Interaction

        class _UnsupportedInteraction(PICMI_Interaction):
            pass

        sim, _, _ = _interaction_sim()
        with pytest.raises(ValueError, match="not .* implemented by PIConGPU|not supported by PIConGPU"):
            sim.add_interaction(_UnsupportedInteraction())

    def test_standard_base_field_ionization_keldysh(self):
        """a plain picmistandard.PICMI_FieldIonization is accepted and mapped to the concrete model"""
        sim, ion, e = _interaction_sim()
        standard = picmistandard.PICMI_FieldIonization(model="Keldysh", ionized_species=ion, product_species=e)
        # the base class is not an instance of the PIConGPU adapter
        assert not isinstance(standard, picmi.FieldIonization)

        sim.add_interaction(standard)

        model = sim.interactions[0]
        assert isinstance(model, Keldysh)
        assert model.ion_species is ion
        assert model.ionization_electron_species is e
        rendered = _render_species_definition(sim)
        assert "Keldysh" in rendered

    def test_standard_base_field_ionization_adk_needs_variant(self):
        """a plain standard ADK request raises the actionable ADK_variant error (no knob on the standard object)"""
        sim, ion, e = _interaction_sim()
        standard = picmistandard.PICMI_FieldIonization(model="ADK", ionized_species=ion, product_species=e)
        with pytest.raises(ValueError, match="ADK_variant"):
            sim.add_interaction(standard)

    def test_standard_base_field_ionization_unknown_model(self):
        """a plain standard object with an unsupported model raises the clear model error"""
        sim, ion, e = _interaction_sim()
        standard = picmistandard.PICMI_FieldIonization(model="ThomasFermi", ionized_species=ion, product_species=e)
        with pytest.raises(ValueError, match="Unsupported field ionization model"):
            sim.add_interaction(standard)

    def test_plain_bsi_with_empty_extensions_renders(self):
        """BSI_extensions=() selects the plain BSI model without extensions"""
        sim, ion, e = _interaction_sim()
        sim.add_interaction(
            picmi.FieldIonization(model="BSI", ionized_species=ion, product_species=e, BSI_extensions=())
        )
        model = sim.interactions[0]
        assert isinstance(model, BSI)
        assert model.BSI_extensions == ()
        assert "BSI" in _render_species_definition(sim)

    def test_irrelevant_knobs_are_rejected(self):
        """a knob that does not belong to the selected model is rejected, not silently ignored"""
        sim, ion, e = _interaction_sim()
        with pytest.raises(ValueError, match="ADK_variant is only valid for the ADK model"):
            sim.add_interaction(
                picmi.FieldIonization(
                    model="Keldysh",
                    ionized_species=ion,
                    product_species=e,
                    ADK_variant=ADKVariant.LinearPolarization,
                )
            )
        sim, ion, e = _interaction_sim()
        with pytest.raises(ValueError, match="BSI_extensions is only valid for the BSI model"):
            sim.add_interaction(
                picmi.FieldIonization(
                    model="Keldysh",
                    ionized_species=ion,
                    product_species=e,
                    BSI_extensions=[BSIExtension.StarkShift],
                )
            )

    def test_unknown_model_raises(self):
        sim, ion, e = _interaction_sim()
        with pytest.raises(ValueError, match="Unsupported field ionization model"):
            sim.add_interaction(picmi.FieldIonization(model="ThomasFermi", ionized_species=ion, product_species=e))
