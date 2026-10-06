"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import tempfile
import warnings
from pathlib import Path

import pytest

from picongpu import core, picmi


def _grid(**kwargs) -> picmi.Cartesian3DGrid:
    base = dict(
        number_of_cells=[192, 2048, 64],
        lower_bound=[0, 0, 0],
        upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-5],
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )
    base.update(kwargs)
    return picmi.Cartesian3DGrid(**base)


def _sim(grid) -> picmi.Simulation:
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    return picmi.Simulation(time_step_size=1.39e-16, max_steps=32, solver=solver)


def test_default_keeps_cpp_defaults():
    """without configuration the pypicongpu default absorber (== static C++ file) is used"""
    absorber = _grid().get_as_pypicongpu().field_absorber
    assert absorber.kind == "pml"
    assert absorber.thickness == ((12, 12), (12, 12), (12, 12))
    assert absorber.strength == ((1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3))


def test_pml_cells_symmetric():
    """the standard pml_cells maps to the per-axis symmetric NUM_CELLS"""
    absorber = _grid(pml_cells=[12, 12, 12]).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((12, 12), (12, 12), (12, 12))


def test_pml_cells_per_axis_values():
    absorber = _grid(pml_cells=[13, 12, 11]).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((13, 13), (12, 12), (11, 11))


def test_pml_cells_disables_axis_with_zero():
    """pml_cells=0 on an axis disables absorption there (thickness 0)"""
    absorber = _grid(pml_cells=[13, 0, 11]).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((13, 13), (0, 0), (11, 11))


def test_kind_extension():
    """the picongpu_absorber_kind extension selects the --fieldAbsorber profile"""
    absorber = _grid(picongpu_absorber_kind="exponential").get_as_pypicongpu().field_absorber
    assert absorber.kind == "exponential"


def test_per_direction_depth_extension():
    """the picongpu_pml_cells extension exposes the full per-direction [3][2] depth"""
    absorber = _grid(picongpu_pml_cells=[[13, 0], [4, 12], [32, 32]]).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((13, 0), (4, 12), (32, 32))


def test_exponential_strength_extension():
    """the picongpu_exponential_strength extension exposes exponential::STRENGTH"""
    absorber = (
        _grid(
            picongpu_absorber_kind="exponential",
            picongpu_exponential_strength=[[1e-2, 2e-2], [1e-3, 1e-3], [1e-3, 2.5e-3]],
        )
        .get_as_pypicongpu()
        .field_absorber
    )
    assert absorber.strength == ((1e-2, 2e-2), (1e-3, 1e-3), (1e-3, 2.5e-3))


def test_pml_cells_and_extension_mutually_exclusive():
    with pytest.raises(ValueError, match=".*only one of them.*"):
        _grid(pml_cells=[12, 12, 12], picongpu_pml_cells=[[12, 12], [12, 12], [12, 12]]).get_as_pypicongpu()


def test_invalid_pml_cells():
    with pytest.raises(ValueError, match=".*pml_cells must be a list of 3.*"):
        _grid(pml_cells=[12, 12]).get_as_pypicongpu()
    with pytest.raises(ValueError, match=".*pml_cells.*"):
        _grid(pml_cells=[12, -2, 12]).get_as_pypicongpu()


def test_domain_fit_hard_error():
    """an explicitly configured absorber that does not fit is a hard Python error"""
    grid = _grid(number_of_cells=[16, 2048, 64], pml_cells=[10, 12, 12])
    with pytest.raises(ValueError, match=".*field absorber in x direction does not fit.*"):
        grid.get_as_pypicongpu()


def test_domain_fit_error_per_direction():
    """the per-direction extension is checked against the boundary devices"""
    grid = _grid(number_of_cells=[80, 2048, 64], picongpu_pml_cells=[[0, 100], [12, 12], [12, 12]])
    with pytest.raises(ValueError, match=".*field absorber in x direction does not fit.*"):
        grid.get_as_pypicongpu()


def test_profile_only_is_not_a_depth_choice():
    """selecting only a profile (kind or strength) must not trigger the depth fit error"""
    # small domain: the default depth of 12 per side would not fit
    grid = _grid(number_of_cells=[16, 16, 16], picongpu_absorber_kind="exponential")
    grid.get_as_pypicongpu()

    grid = _grid(
        number_of_cells=[16, 16, 16],
        picongpu_exponential_strength=[[1e-3, 1e-3], [1e-3, 1e-3], [1e-3, 1e-3]],
    )
    grid.get_as_pypicongpu()


def test_kind_only_does_not_warn_about_default_thickness_on_periodic_axis():
    """picking a profile without a depth must not warn about the default thickness"""
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[40, 40, 40],
        lower_bound=[0, 0, 0],
        upper_bound=[4e-5, 4e-5, 4e-5],
        lower_boundary_conditions=["periodic", "open", "open"],
        upper_boundary_conditions=["periodic", "open", "open"],
        picongpu_absorber_kind="exponential",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid.get_as_pypicongpu()


def test_warning_points_at_caller():
    """the periodic-axis warning must not point into the library internals"""
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[40, 40, 40],
        lower_bound=[0, 0, 0],
        upper_bound=[4e-5, 4e-5, 4e-5],
        lower_boundary_conditions=["periodic", "open", "open"],
        upper_boundary_conditions=["periodic", "open", "open"],
        pml_cells=[12, 12, 12],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid.get_as_pypicongpu()
    assert caught
    assert caught[0].filename == __file__


def test_periodic_axis_is_noop_warning():
    """a thickness on a periodic axis is a warning (PIConGPU applies no absorber there)"""
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[192, 2048, 64],
        lower_bound=[0, 0, 0],
        upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-5],
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
        pml_cells=[12, 12, 12],
    )
    with pytest.warns(UserWarning, match=".*periodic z axis is ignored.*"):
        grid.get_as_pypicongpu()


def test_all_periodic_absorber_off_warning():
    """on an all-periodic grid the C++ core forces the absorber kind to None"""
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[192, 2048, 64],
        lower_bound=[0, 0, 0],
        upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-5],
        lower_boundary_conditions=["periodic", "periodic", "periodic"],
        upper_boundary_conditions=["periodic", "periodic", "periodic"],
        pml_cells=[12, 12, 12],
    )
    with pytest.warns(UserWarning, match=".*All boundaries are periodic.*"):
        grid.get_as_pypicongpu()


def test_default_write_input_file_byte_equal():
    """full setup generation renders fieldAbsorber.param byte-equal to the static C++ file"""
    static = (core.path("include") / "picongpu/param/fieldAbsorber.param").read_bytes()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "setup"
        _sim(_grid()).write_input_file(outdir)
        rendered = (outdir / "include/picongpu/param/fieldAbsorber.param").read_bytes()
        assert rendered == static
        cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert 'TBG_fieldAbsorber="--fieldAbsorber pml"' in cfg


def test_exponential_kind_wired_into_cfg():
    """kind selection ends up in the --fieldAbsorber command line option"""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "setup"
        _sim(_grid(picongpu_absorber_kind="exponential")).write_input_file(outdir)
        cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert 'TBG_fieldAbsorber="--fieldAbsorber exponential"' in cfg
        rendered = (outdir / "include/picongpu/param/fieldAbsorber.param").read_text()
        assert "namespace exponential" in rendered
        assert "constexpr float_X STRENGTH[3][2]" in rendered


def test_2d_pml_cells_mapping():
    """the standard pml_cells in 2D has two entries; the third (inert) axis keeps the default"""
    grid = picmi.Cartesian2DGrid(
        number_of_cells=[128, 128],
        lower_bound=[0, 0],
        upper_bound=[0.064, 0.064],
        lower_boundary_conditions=["open", "open"],
        upper_boundary_conditions=["open", "open"],
        pml_cells=[13, 11],
    )
    absorber = grid.get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((13, 13), (11, 11), (12, 12))


def test_2d_wrong_pml_cells_length_rejected():
    grid = picmi.Cartesian2DGrid(
        number_of_cells=[128, 128],
        lower_bound=[0, 0],
        upper_bound=[0.064, 0.064],
        lower_boundary_conditions=["open", "open"],
        upper_boundary_conditions=["open", "open"],
        pml_cells=[13, 11, 12],
    )
    with pytest.raises(ValueError, match=".*pml_cells must be a list of 2.*"):
        grid.get_as_pypicongpu()


def test_2d_domain_fit_hard_error():
    grid = picmi.Cartesian2DGrid(
        number_of_cells=[16, 128],
        lower_bound=[0, 0],
        upper_bound=[0.008, 0.064],
        lower_boundary_conditions=["open", "open"],
        upper_boundary_conditions=["open", "open"],
        pml_cells=[10, 12],
    )
    with pytest.raises(ValueError, match=".*field absorber in x direction does not fit.*"):
        grid.get_as_pypicongpu()
