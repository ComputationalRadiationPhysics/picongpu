"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from picongpu import rc_params
from picongpu._rc_params import RCParams
from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation
from picongpu.pypicongpu import runner
from picongpu.pypicongpu.runner import (
    PicBuildFlags,
    Runner,
    TBGFlags,
    generate_bare_profile,
    generate_bare_profile_as_in,
)
from picongpu.pypicongpu.util import UnpackChain
from pydantic import ValidationError
from pytest import fixture, raises


@fixture
def empty_rc_params():
    return type(rc_params)()


@fixture
def picmi_sim():
    number_of_cells = 32
    return Simulation(
        time_step_size=17,
        max_steps=4,
        solver=ElectromagneticSolver(
            method="Yee",
            grid=Cartesian3DGrid(
                number_of_cells=[number_of_cells] * 3,
                lower_bound=[0, 0, 0],
                upper_bound=[number_of_cells] * 3,
                # required, otherwise won't spawn
                lower_boundary_conditions=["open", "open", "periodic"],
                upper_boundary_conditions=["open", "open", "periodic"],
            ),
        ),
    )


@fixture
def arbitrary_string():
    return "Hello World"


def test_generate_bare_profile(empty_rc_params, arbitrary_string):
    empty_rc_params["profile_content"] = arbitrary_string
    with NamedTemporaryFile(mode="r") as file:
        result_path = generate_bare_profile(path=file.name, rc_params=empty_rc_params)
        assert arbitrary_string in file.read()
        assert Path(file.name) == result_path


def test_generate_bare_profile_as_in(arbitrary_string):
    with NamedTemporaryFile(mode="w", suffix=".py") as script:
        script.write(
            f"""
from picongpu import rc_params

rc_params["profile_content"] = "{arbitrary_string}"
"""
        )
        script.flush()
        with NamedTemporaryFile(mode="r") as file:
            result_path = generate_bare_profile_as_in(script_path=script.name, path=file.name)
            assert arbitrary_string in file.read()
            assert Path(file.name) == result_path


def test_init_args_take_precedence_over_picongpurc():
    with NamedTemporaryFile("w") as file:
        file.write('author = "a"')
        file.flush()
        assert RCParams(author="b", picongpurc_path=Path(file.name))["author"] == "b"


def test_init_args_take_precedence_over_preset(arbitrary_string):
    # just to be sure the test isn't trivial:
    assert RCParams(preset="bash")["pic_backend"] != arbitrary_string

    assert RCParams(pic_backend=arbitrary_string, preset="bash")["pic_backend"] == arbitrary_string


def test_picongpurc_content_takes_precedence_over_preset(arbitrary_string):
    # just to be sure the test isn't trivial:
    assert RCParams(preset="bash")["pic_backend"] != arbitrary_string

    with NamedTemporaryFile("w") as file:
        file.write(f'pic_backend = "{arbitrary_string}"')
        file.flush()
        assert RCParams(preset="bash", picongpurc_path=Path(file.name))["pic_backend"] == arbitrary_string


def test_picbuild_and_tbg_flags_are_disjoint_enough():
    # The overlapping flags mean roughly the same in both, so I consider this fine.
    assert set.intersection(
        *(
            # This only covers the case that I use an AliasChoices object as validation_alias.
            # But that's fine becuase I've done so.
            reduce(set.union, UnpackChain(cls.model_fields).values().validation_alias.choices, set())
            for cls in (PicBuildFlags, TBGFlags)
        )
    ) == {"f", "force"}


def _rc_params_with_picongpurc(monkeypatch, content):
    with TemporaryDirectory() as directory:
        picongpurc_path = Path(directory) / "picongpurc.toml"
        picongpurc_path.write_text(content)
        config = RCParams(picongpurc_path=picongpurc_path)
    monkeypatch.setattr(runner, "rc_params", config)
    monkeypatch.setattr("picongpu.rc_params", config)
    return config


def test_build_jobs_default_reads_from_picongpurc(monkeypatch):
    _rc_params_with_picongpurc(monkeypatch, "build_jobs = 8\n")
    assert PicBuildFlags().jobs == 8


def test_build_jobs_default_falls_back_to_four(monkeypatch):
    _rc_params_with_picongpurc(monkeypatch, "")
    assert PicBuildFlags().jobs == 4


def test_explicit_jobs_overrides_picongpurc_build_jobs(monkeypatch):
    _rc_params_with_picongpurc(monkeypatch, "build_jobs = 8\n")
    assert PicBuildFlags(jobs=16).jobs == 16


def test_build_jobs_none_means_infinite_jobs(monkeypatch):
    # TOML does not know a `null` literal, so this state can only be
    # reached programmatically, e.g. via plugin or Python API.
    config = _rc_params_with_picongpurc(monkeypatch, "")
    config["build_jobs"] = None
    assert PicBuildFlags().jobs is None


def test_string_build_jobs_is_coerced_to_int(monkeypatch):
    # `default_factory` results are validated like explicit args, so a
    # mis-typed string `build_jobs` is coerced (or rejected) instead of
    # silently landing in input.yaml (where it would only fail at the CWL layer).
    _rc_params_with_picongpurc(monkeypatch, 'build_jobs = "8"\n')
    assert PicBuildFlags().jobs == 8


def test_unparseable_build_jobs_is_rejected(monkeypatch):
    _rc_params_with_picongpurc(monkeypatch, 'build_jobs = "many"\n')
    with raises(ValidationError):
        PicBuildFlags().jobs


def test_float_build_jobs_is_rejected(monkeypatch):
    _rc_params_with_picongpurc(monkeypatch, "build_jobs = 8.5\n")
    with raises(ValidationError):
        PicBuildFlags().jobs


def test_generate_exist_ok_regenerates_existing_setup_dir(picmi_sim, tmp_path):
    """generate(exist_ok=True) overwrites a previously generated setup dir instead of failing (#5752)"""
    setup_dir = tmp_path / "setup"
    runner = Runner(sim=picmi_sim, setup_dir=setup_dir)
    runner.generate()

    param_file = setup_dir / "include" / "picongpu" / "param" / "simulation.param"
    nested_file = setup_dir / "etc" / "picongpu" / "N.cfg"
    assert param_file.is_file()
    assert nested_file.is_file()
    expected_param_content = param_file.read_text()
    expected_nested_content = nested_file.read_text()

    # simulate stale/modified output from the previous generation:
    # both a nested template dir file and a default template file
    param_file.write_text("stale")
    nested_file.write_text("stale")

    runner.generate(exist_ok=True)

    assert param_file.read_text() == expected_param_content
    assert nested_file.read_text() == expected_nested_content
