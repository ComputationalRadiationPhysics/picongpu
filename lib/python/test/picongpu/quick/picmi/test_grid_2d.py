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


def _render_template(grid, template_name: str) -> str:
    """Render a single ``.mustache`` template from the production rendering context.

    Uses the production rendering context + Renderer (no full directory copy), so a
    single template can be asserted cheaply in the quick suite.
    """
    sim = picmi.Simulation(max_steps=1, solver=picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid))
    pypic = sim.get_as_pypicongpu()
    context = pypic.get_rendering_context()
    Renderer.check_rendering_context(context)
    preprocessed = Renderer.get_context_preprocessed(context)
    template = (templates.path() / "include" / "picongpu" / "param" / template_name).read_text()
    return Renderer.get_rendered_template(preprocessed, template)


def render_memory_param_supercell_size(grid) -> str:
    rendered = _render_template(grid, "memory.param.mustache")
    return next(line.strip() for line in rendered.splitlines() if "SuperCellSize =" in line)


def render_memory_param_guard_size(grid) -> str:
    """Render just the memory.param template and return the GuardSize line (see render_memory_param_supercell_size)."""
    sim = picmi.Simulation(max_steps=1, solver=picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid))
    pypic = sim.get_as_pypicongpu()
    context = pypic.get_rendering_context()
    Renderer.check_rendering_context(context)
    preprocessed = Renderer.get_context_preprocessed(context)
    template = (templates.path() / "include" / "picongpu" / "param" / "memory.param.mustache").read_text()
    rendered = Renderer.get_rendered_template(preprocessed, template)
    return next(line.strip() for line in rendered.splitlines() if "GuardSize =" in line)


def get_sim_cfl_2d(delta_t, cfl, delta_2d, method="Yee", n=100) -> picmi.Simulation:
    grid = get_grid_2d(delta_2d[0], delta_2d[1], n)
    solver = picmi.ElectromagneticSolver(method=method, grid=grid, cfl=cfl)
    return picmi.Simulation(time_step_size=delta_t, solver=solver)


def render_cell_depth_si(grid) -> float:
    rendered = _render_template(grid, "simulation.param.mustache")
    line = next(line for line in rendered.splitlines() if "CELL_DEPTH_SI =" in line)
    return float(line.split("CELL_DEPTH_SI = ")[1].rstrip(";").strip())


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
        # 2D: plain 2-component Int (no simDim, no shrinkTo). The 2D super-cell
        # default is <16, 16>, matching PIConGPU's 2D setups (e.g. the FoilLCT example).
        assert line_2d == "using SuperCellSize = mCT::Int<16, 16>;"

    def test_memory_param_guard_size_well_formed(self):
        # Regression guard: 2D must support guard cells with the same semantics as 3D
        # (guard_cells in cells -> guard_size in super cells). Previously the 2D grid
        # validated guard_cells but silently dropped them (no guard_size conversion), so
        # e.g. guard_cells=[32,32] on a <16,16> super cell rendered the default <1,1,1>.
        base = dict(
            number_of_cells=[128, 128],
            lower_bound=[0, 0],
            upper_bound=[0.064, 0.064],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
            picongpu_super_cell_size=[16, 16],
        )
        # 2D: no guard_cells -> default; guard_cells are mapped to super-cell counts.
        # The 2D z placeholder is 1 and is dropped by shrinkTo<..., simDim> in 2D3V.
        assert render_memory_param_guard_size(picmi.Cartesian2DGrid(**base)) == (
            "using GuardSize = typename mCT::shrinkTo<mCT::Int<1, 1, 1>, simDim>::type;"
        )
        assert render_memory_param_guard_size(picmi.Cartesian2DGrid(**base, guard_cells=[32, 32])) == (
            "using GuardSize = typename mCT::shrinkTo<mCT::Int<2, 2, 1>, simDim>::type;"
        )
        # conversion: guard_size in super cells = guard_cells // super_cell_size
        assert picmi.Cartesian2DGrid(**base, guard_cells=[32, 16]).get_as_pypicongpu().guard_size == (2, 1)
        # 3D unchanged: same line, z component carried (super cell default <8,8,4>)
        assert (
            render_memory_param_guard_size(
                picmi.Cartesian3DGrid(
                    number_of_cells=[32, 32, 32],
                    lower_bound=[0, 0, 0],
                    upper_bound=[1e-6, 1e-6, 1e-6],
                    lower_boundary_conditions=["open", "open", "open"],
                    upper_boundary_conditions=["open", "open", "open"],
                )
            )
            == "using GuardSize = typename mCT::shrinkTo<mCT::Int<1, 1, 1>, simDim>::type;"
        )

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

    def test_cell_depth_si_explicit_override(self):
        # An explicit picongpu_cell_depth_si must be rendered as CELL_DEPTH_SI,
        # overriding the dx default (2D3V slab thickness).
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[128, 128],
            lower_bound=[0, 0],
            upper_bound=[0.064, 0.064],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
            picongpu_cell_depth_si=1.5e-6,
        )
        assert render_cell_depth_si(grid) == pytest.approx(1.5e-6)

    def test_cell_depth_si_defaults_to_dx(self):
        # When picongpu_cell_depth_si is left as None, CELL_DEPTH_SI falls back
        # to the x cell size (dx), preserving the historical behaviour.
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[128, 128],
            lower_bound=[0, 0],
            upper_bound=[0.064, 0.064],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        dx = 0.064 / 128
        assert render_cell_depth_si(grid) == pytest.approx(dx)

    def test_cell_depth_si_rejects_non_positive(self):
        grid_kwargs = dict(
            number_of_cells=[128, 128],
            lower_bound=[0, 0],
            upper_bound=[0.064, 0.064],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        for bad in (0.0, -1.5e-6):
            with pytest.raises(ValueError, match="cell depth"):
                picmi.Cartesian2DGrid(**grid_kwargs, picongpu_cell_depth_si=bad)


def get_grid_3d(delta_x: float, delta_y: float, delta_z: float, n: int = 100):
    # sets delta_[x,y,z] implicitly by providing bounding box + cell count
    return picmi.Cartesian3DGrid(
        number_of_cells=[n, n, n],
        lower_bound=[0, 0, 0],
        upper_bound=[n * delta_x, n * delta_y, n * delta_z],
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )


class TestCartesian3DGridTo2D(TestCase):
    def test_to_2d_drops_z_component(self):
        # Every vector field is mapped from a 3-tuple to a 2-tuple by keeping
        # the (x, y) components; z is dropped. A fresh 2D grid is returned and
        # the source 3D grid is left untouched.
        delta = 1e-6
        grid_3d = get_grid_3d(delta, 2e-6, 4e-6, n=128)
        grid_2d = grid_3d.to_2d()

        assert isinstance(grid_2d, picmi.Cartesian2DGrid)
        assert len(grid_2d.number_of_cells) == 2
        assert grid_2d.number_of_cells == [128, 128]
        assert grid_2d.lower_bound == [0, 0]
        assert grid_2d.upper_bound == [128 * delta, 128 * 2e-6]
        assert len(grid_2d.lower_boundary_conditions) == 2
        assert len(grid_2d.upper_boundary_conditions) == 2
        assert len(grid_2d.picongpu_cell_size) == 2

        # the source grid is not mutated
        assert len(grid_3d.number_of_cells) == 3

    def test_to_2d_cell_depth_is_3d_z_cell_size(self):
        # The z cell length becomes the 2D slab thickness (CELL_DEPTH_SI).
        # get_grid_3d uses per-cell sizes, so the z cell size is delta_z.
        delta_z = 4e-6
        grid_3d = get_grid_3d(1e-6, 2e-6, delta_z, n=128)
        grid_2d = grid_3d.to_2d()
        assert grid_2d.picongpu_cell_depth_si == pytest.approx(delta_z)

    def test_to_2d_supercell_default(self):
        # The 3D default super cell (8, 8, 4) maps to the 2D default (16, 16).
        # n=128 so the reduced grid's cell count is a multiple of the 2D default
        # super cell (16), keeping the reduction valid at the 2D level.
        grid_2d = get_grid_3d(1e-6, 1e-6, 1e-6, n=128).to_2d()
        assert grid_2d.picongpu_super_cell_size == (16, 16)

    def test_to_2d_supercell_explicit_keeps_xy(self):
        # An explicitly-set 3D super cell keeps its (x, y) components (z dropped).
        grid_3d = picmi.Cartesian3DGrid(
            number_of_cells=[128, 128, 128],
            lower_bound=[0, 0, 0],
            upper_bound=[128e-6, 128e-6, 128e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
            picongpu_super_cell_size=[16, 8, 4],
        )
        assert grid_3d.to_2d().picongpu_super_cell_size == (16, 8)

    def test_to_2d_supercell_explicit_default_value_keeps_xy(self):
        # An explicit 3D super cell equal to the default value (8, 8, 4) must
        # still keep its (x, y) = (8, 8), not snap to the 2D default (16, 16).
        # Default-vs-explicit is detected via model_fields_set, not the value.
        grid_3d = picmi.Cartesian3DGrid(
            number_of_cells=[16, 16, 128],
            lower_bound=[0, 0, 0],
            upper_bound=[16e-6, 16e-6, 128e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
            picongpu_super_cell_size=[8, 8, 4],
        )
        assert grid_3d.to_2d().picongpu_super_cell_size == (8, 8)

    def test_to_2d_maps_gpu_and_guard(self):
        # n_gpus, guard_cells and grid_dist drop the z (third) component.
        grid_3d = picmi.Cartesian3DGrid(
            number_of_cells=[64, 128, 4],
            lower_bound=[0, 0, 0],
            upper_bound=[64e-6, 128e-6, 4e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
            picongpu_n_gpus=[1, 2, 1],
            picongpu_super_cell_size=[8, 8, 4],
            guard_cells=[16, 16, 4],
            picongpu_grid_dist=[(64,), (64, 64), (4,)],
        )
        grid_2d = grid_3d.to_2d()
        assert grid_2d.picongpu_n_gpus == (1, 2)
        assert grid_2d.guard_cells == [16, 16]
        assert grid_2d.picongpu_grid_dist == [[64], [64, 64]]
        assert grid_2d.get_as_pypicongpu().guard_size == (2, 2)

    def test_to_2d_passes_2d_validation(self):
        # The returned grid is a fully valid Cartesian2DGrid: converting it to
        # pypicongpu (which runs the 2D validators) must succeed and render a
        # 2D grid with the 3D z cell size as the slab thickness.
        delta_z = 4e-6
        grid_2d = get_grid_3d(1e-6, 2e-6, delta_z, n=128).to_2d()
        pypic = grid_2d.get_as_pypicongpu()
        assert pypic.sim_dim == 2
        assert pypic.has_z is False
        assert pypic.cell_depth == pytest.approx(delta_z)

    def test_to_2d_invalid_reduction_raises(self):
        # The reduction is validated against the 2D constraints (spec Q6). A 3D
        # grid that is valid in 3D but whose reduction violates them must raise
        # at the call site instead of returning an invalid grid: here the 3D
        # default super cell snaps to the 2D (16, 16) while the 3D-valid guard
        # (a multiple of 8, not 16) does not divide it.
        grid_3d = picmi.Cartesian3DGrid(
            number_of_cells=[128, 128, 4],
            lower_bound=[0, 0, 0],
            upper_bound=[128e-6, 128e-6, 4e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
            guard_cells=[8, 8, 4],
        )
        grid_3d.check()
        with pytest.raises(ValueError, match="super cell size"):
            grid_3d.to_2d()
