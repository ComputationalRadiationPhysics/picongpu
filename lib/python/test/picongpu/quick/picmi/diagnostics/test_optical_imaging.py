"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import os
import tempfile
from unittest import TestCase

import sympy

from picongpu import picmi
from picongpu.picmi.diagnostics import OpticalImaging, Shadowgraphy


def _always_one(*_):
    return 1.0


def _unit_imaging(**overrides):
    kwargs = dict(
        duration=12,
        position_wf=_always_one,
        time_wf=_always_one,
        mask_fourier=_always_one,
    )
    kwargs.update(overrides)
    return OpticalImaging(**kwargs)


def _simulation(diagnostics, grid=None):
    grid = grid or picmi.Cartesian3DGrid(
        number_of_cells=[32, 32, 64],
        lower_bound=[0, 0, 0],
        upper_bound=[3.2e-6, 3.2e-6, 6.4e-6],
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid, cfl=1.0)
    sim = picmi.Simulation(max_steps=100, solver=solver)
    electrons = picmi.Species(
        name="e", particle_type="electron", initial_distribution=picmi.UniformDistribution(density=1e20)
    )
    sim.add_species(electrons, picmi.PseudoRandomLayout(n_macroparticles_per_cell=2))
    sim.diagnostics = diagnostics
    return sim


def _render(sim):
    with tempfile.TemporaryDirectory() as outdir:
        sim.write_input_file(outdir, exist_ok=True)
        n_cfg = os.path.join(outdir, "etc", "picongpu", "N.cfg")
        with open(n_cfg) as cfg:
            n_cfg_content = cfg.read()
        param = os.path.join(outdir, "include", "picongpu", "param", "shadowgraphy.param")
        with open(param) as param_file:
            param_content = param_file.read()
    return n_cfg_content, param_content


class TestOpticalImagingConstruction(TestCase):
    def test_duration_rounded_to_multiple_of_t_res(self):
        assert Shadowgraphy(duration=13, t_res=3).duration == 12

    def test_duration_below_one_t_res_rejected(self):
        with self.assertRaises(ValueError):
            Shadowgraphy(duration=2, t_res=3)

    def test_slice_point_one_rejected(self):
        with self.assertRaises(ValueError):
            _unit_imaging(slice_point=1.0)

    def test_optical_imaging_has_no_mask_defaults(self):
        with self.assertRaises(ValueError):
            OpticalImaging(duration=4)

    def test_shadowgraphy_preset_enables_final_output(self):
        assert Shadowgraphy(duration=4).final_output is True

    def test_mismatched_mask_symbol_rejected(self):
        def bad(kx, ky, omega):
            return kx * sympy.Symbol("not_a_param")

        with self.assertRaises(ValueError):
            _unit_imaging(mask_fourier=bad).get_as_pypicongpu()


class TestOpticalImagingRendering(TestCase):
    def test_n_cfg_options_rendered_per_instance(self):
        sim = _simulation(
            [
                Shadowgraphy(duration=30, file="sg1", slice_point=0.5),
                Shadowgraphy(
                    duration=12,
                    start=5,
                    file="sg2",
                    slice_point=0.25,
                    focus_pos=1e-3,
                    fourier_output=True,
                    intermediate_output=True,
                ),
            ]
        )
        n_cfg, _ = _render(sim)

        for needle in (
            "--shadowgraphy.duration 30",
            "--shadowgraphy.file sg1",
            "--shadowgraphy.slicePoint 0.5",
            "--shadowgraphy.finalOutput true",
            "--shadowgraphy.start 5",
            "--shadowgraphy.duration 12",
            "--shadowgraphy.file sg2",
            "--shadowgraphy.slicePoint 0.25",
            "--shadowgraphy.focusPos 0.001",
            "--shadowgraphy.fourierOutput true",
            "--shadowgraphy.intermediateOutput true",
        ):
            assert needle in n_cfg, f"{needle!r} not found in rendered N.cfg"

    def test_shadowgraphy_param_rendered(self):
        _, param = _render(_simulation([Shadowgraphy(duration=30)]))
        assert "constexpr unsigned int tRes = 2;" in param
        assert "constexpr float numericalAperture = 0.23000000000000001;" in param
        # the preset renders the Tukey window and Fourier mask C++ code
        assert "HINLINE float_64 positionWf(int i, int j, int pluginNumX, int pluginNumY)" in param
        assert "pmacc::math::cos" in param

    def test_constant_mask_for_plain_optical_imaging(self):
        _, param = _render(_simulation([_unit_imaging()]))
        assert "return 1.0;" in param

    def test_inconsistent_compile_time_values_rejected(self):
        sim = _simulation(
            [
                _unit_imaging(t_res=2),
                _unit_imaging(t_res=4, duration=12),
            ]
        )
        with self.assertRaises(ValueError):
            sim.get_as_pypicongpu()

    def test_no_imaging_renders_default_param(self):
        _, param = _render(_simulation([]))
        assert "constexpr unsigned int tRes = 2;" in param
        assert "return 1.0;" in param


class TestOpticalImagingTwoDimensional(TestCase):
    def test_2d_grid_rejected(self):
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[32, 32],
            lower_bound=[0, 0],
            upper_bound=[1e-6, 1e-6],
            lower_boundary_conditions=["open", "open"],
            upper_boundary_conditions=["open", "open"],
        )
        sim = _simulation([Shadowgraphy(duration=10)], grid=grid)
        with self.assertRaises(ValueError):
            sim.get_as_pypicongpu()
