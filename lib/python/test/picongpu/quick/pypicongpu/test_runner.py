"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile

from picongpu import rc_params
from picongpu._rc_params import RCParams
from picongpu.pypicongpu.runner import (
    PicBuildFlags,
    TBGFlags,
    generate_bare_profile,
    generate_bare_profile_as_in,
)
from picongpu.pypicongpu.util import UnpackChain
from pytest import fixture


@fixture
def my_rc_params():
    return type(rc_params)()


@fixture
def arbitrary_string():
    return "Hello World"


def test_generate_bare_profile(my_rc_params, arbitrary_string):
    my_rc_params["profile_content"] = arbitrary_string
    with NamedTemporaryFile(mode="r") as file:
        result_path = generate_bare_profile(path=file.name, rc_params=my_rc_params)
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
