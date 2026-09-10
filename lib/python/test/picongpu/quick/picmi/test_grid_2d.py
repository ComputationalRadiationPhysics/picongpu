"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import math
from unittest import TestCase

import pytest
from picongpu import picmi
from picongpu import templates
from picongpu.picmi import constants
from picongpu.picmi.diagnostics import PhaseSpace, TimeStepSpec
from picongpu.pypicongpu.rendering.renderer import Renderer


def get_grid_2d(delta_x: float, delta_y: float, n: int = 100):
    # sets delta_[x,y] implicitly by providing bounding box + cell count
    return picmi.Cartesian2DGrid(
        number_of_cells=[n, n],
        lower_bound=[0, 0],
        upper_bound=[n * delta_x, n * delta_y],
        lower_boundary_conditions=["open", "open"],
        upper_boundary_conditions=["open", "open"],
    )


def render_memory_param_supercell_size(grid) -> str:
    """Render just the memory.param template and return the SuperCellSize line.

    Uses the production rendering context + Renderer (no full directory copy), so a
    single template can be asserted cheaply in the quick suite.
    """
    sim = picmi.Simulation(max_steps=1, solver=picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid))
    pypic = sim.get_as_pypicongpu()
    context = pypic.get_rendering_context()
    Renderer.check_rendering_context(context)
    preprocessed = Renderer.get_context_preprocessed(context)
    template = (templates.path() / "include" / "picongpu" / "param" / "memory.param.mustache").read_text()
    rendered = Renderer.get_rendered_template(preprocessed, template)
    return next(line.strip() for line in rendered.splitlines() if "SuperCellSize =" in line)


def get_sim_cfl_2d(delta_t, cfl, delta_2d, method="Yee", n=100) -> picmi.Simulation:
    grid = get_grid_2d(delta_2d[0], delta_2d[1], n)
    solver = picmi.ElectromagneticSolver(method=method, grid=grid, cfl=cfl)
    return picmi.Simulation(time_step_size=delta_t, solver=solver)


class TestCartesian2DGrid(TestCase):
    def test_basic_translation(self):
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[128, 128],
            lower_bound=[0, 0],
            upper_bound=[0.064, 0.064],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        pypic = grid.get_as_pypicongpu()
        context = pypic.get_rendering_context()
        assert context["sim_dim"] == 2
        assert context["has_z"] is False
        assert set(context["cell_cnt"]) == {"x", "y"}
        assert set(context["cell_size"]) == {"x", "y"}
        assert "z" not in context["gpu_cnt"]
        assert "z" not in context["boundary_condition"]

    def test_grid_dist(self):
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[128, 128],
            picongpu_n_gpus=[2, 1],
            picongpu_super_cell_size=[16, 16],
            picongpu_grid_dist=([64, 64], [128]),
            lower_bound=[0, 0],
            upper_bound=[0.064, 0.064],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        pypic = grid.get_as_pypicongpu()
        assert pypic.grid_dist == ([64, 64], [128])

    def test_2d_cfl_uses_2d_factor(self):
        # The 2D CFL factor is sqrt(1/dx^2 + 1/dy^2), i.e. sqrt(2) for a square
        # grid -- NOT the 3D factor sqrt(3). A 2D setup must not silently be given
        # the 3D factor.
        delta = 7e-6
        n = 100
        c = constants.c
        sqrt2 = math.sqrt(1 / delta**2 + 1 / delta**2)  # == sqrt(2)/delta
        sqrt3 = math.sqrt(1 / delta**2 + 1 / delta**2 + 1 / delta**2)  # == sqrt(3)/delta

        # delta_t -> cfl
        delta_t = 2.02760320328617635877e-13
        sim = get_sim_cfl_2d(delta_t, None, (delta, delta), "Yee", n=n)
        expected_cfl = delta_t * c * sqrt2
        assert abs(sim.solver.cfl - expected_cfl) < 1e-6 * expected_cfl
        assert abs(sim.solver.cfl - delta_t * c * sqrt3) > 1e-3 * expected_cfl

        # cfl -> delta_t
        cfl = 0.99
        sim = get_sim_cfl_2d(None, cfl, (delta, delta), "Yee", n=n)
        expected_dt = cfl / (c * sqrt2)
        assert abs(sim.time_step_size - expected_dt) < 1e-20
        assert abs(sim.time_step_size - cfl / (c * sqrt3)) > 0.01 * expected_dt

        # both given & matching -> ok
        get_sim_cfl_2d(expected_dt, cfl, (delta, delta), "Yee", n=n)
        # both given & mismatched -> raises
        with pytest.raises(ValueError):
            get_sim_cfl_2d(1.0, cfl, (delta, delta), "Yee", n=n)

    def test_3d_cfl_uses_3d_factor(self):
        # Regression guard: a 3D grid must still use the 3D factor sqrt(3)
        # for a square grid.
        delta = 7e-6
        n = 100
        c = constants.c
        delta_t = 2.02760320328617635877e-13
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[n, n, n],
            lower_bound=[0, 0, 0],
            upper_bound=[n * delta, n * delta, n * delta],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=delta_t, solver=solver)
        expected_cfl = delta_t * c * math.sqrt(3) / delta
        assert abs(sim.solver.cfl - expected_cfl) < 1e-6 * expected_cfl

    def test_2d_rejects_phase_space_spatial_z(self):
        # 2D3V: the spatial coordinate must be x or y (no z); momentum keeps pz.
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[64, 64],
            lower_bound=[0, 0],
            upper_bound=[0.032, 0.032],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)
        sim = picmi.Simulation(max_steps=1, solver=solver)
        electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            initial_distribution=picmi.UniformDistribution(density=1.0e24),
        )
        sim.add_species(electrons, layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=1))

        ps = PhaseSpace(
            species=electrons,
            period=TimeStepSpec[::10],
            spatial_coordinate="z",
            momentum_coordinate="py",
            min_momentum=-1.0,
            max_momentum=1.0,
        )
        sim.diagnostics.append(ps)
        with pytest.raises(ValueError, match=".*spatial coordinate.*z.*2D.*"):
            sim.get_as_pypicongpu()

        # a spatial coordinate of x/y is accepted (momentum pz stays valid in 2D3V)
        ps.spatial_coordinate = "x"
        sim.get_as_pypicongpu()

    def test_memory_param_supercell_size_well_formed(self):
        # Regression guard for the 3D memory.param SuperCellSize line: the 3D branch
        # must emit a *well-formed* `shrinkTo<Int<...>, simDim>` (Int closed with `>`),
        # not a stray `}` (a brace-count typo that broke every 3D build).
        grid_3d = picmi.Cartesian3DGrid(
            number_of_cells=[32, 32, 32],
            lower_bound=[0, 0, 0],
            upper_bound=[1e-6, 1e-6, 1e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
        )
        grid_2d = picmi.Cartesian2DGrid(
            number_of_cells=[32, 32],
            lower_bound=[0, 0],
            upper_bound=[1e-6, 1e-6],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )

        line_3d = render_memory_param_supercell_size(grid_3d)
        line_2d = render_memory_param_supercell_size(grid_2d)

        # 3D: shrinkTo<Int<8, 8, 4>, simDim>::type  -- Int closed with `>`, simDim is an
        # argument of shrinkTo, not a 4th Int component. A `}` here is a parse error.
        assert line_3d == "using SuperCellSize = typename mCT::shrinkTo<mCT::Int<8, 8, 4>, simDim>::type;"
        assert "}" not in line_3d
        # 2D: plain 2-component Int (no simDim, no shrinkTo).
        assert line_2d == "using SuperCellSize = mCT::Int<8, 8>;"

    def test_2d_rejects_z_dependent_analytic_density(self):
        # 2D3V has no spatial z coordinate; a z-dependent free-formula density would
        # render an undeclared `z` (C++ compile error). Reject it at validation time.
        from picongpu.picmi import AnalyticDistribution

        grid = picmi.Cartesian2DGrid(
            number_of_cells=[64, 64],
            lower_bound=[0, 0],
            upper_bound=[0.032, 0.032],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)

        def build(density_function):
            sim = picmi.Simulation(max_steps=1, solver=solver)
            species = picmi.Species(
                particle_type="electron",
                name="electrons",
                initial_distribution=AnalyticDistribution(density_function=density_function),
            )
            sim.add_species(species, layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=1))
            return sim

        # z-dependent density on a 2D grid -> clear PICMI error
        with pytest.raises(ValueError, match=".*z.*2D.*"):
            build(lambda x, y, z: x * y * z).get_as_pypicongpu()

        # a 2D-safe (z-independent) density is accepted
        build(lambda x, y, z: x * y).get_as_pypicongpu()
