"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import tomli_w
from picongpu import rc_params
from picongpu._rc_params import find_picongpurc_path
from picongpu.pypicongpu.runner import run_commands
from pytest import fixture, mark, raises


@fixture
def my_rc_params():
    return type(rc_params)()


def test_raises_on_failing_command():
    with raises(RuntimeError):
        run_commands("false")


def test_raises_on_failing_command_with_others_later():
    with raises(RuntimeError):
        run_commands(["false", "true"])


def test_passes_failures_with_overriden_preamble(my_rc_params):
    my_rc_params["preamble"] = ""
    # Does not raise a RuntimeError as above
    # because we have overridden the preamble
    # that applied `set -e` among other things.
    run_commands(["false", "true"], rc_params=my_rc_params)


def test_uses_profile_content(my_rc_params):
    my_rc_params["profile_content"] = "false"
    with raises(RuntimeError):
        run_commands("true", rc_params=my_rc_params)


def test_uses_profile_path(my_rc_params):
    with NamedTemporaryFile("w") as file:
        file.write("false")
        file.flush()
        my_rc_params["profile_path"] = file.name
        with raises(RuntimeError):
            run_commands("true", rc_params=my_rc_params)


# Taking a little guess here that these two will be available on any test system:
@mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
def test_respects_shebang_in_profile(shell, my_rc_params):
    my_rc_params["profile_content"] = f"#!{shell}"
    run_commands(
        ['command="$(ps -p $$ -o cmd | tail -n 1)"', f'[ "${{command% *}}" = {shell} ]'],
        rc_params=my_rc_params,
    )


def test_respects_overriden_shebang(my_rc_params):
    shebang = "/bin/sh"
    my_rc_params["profile_content"] = "#!/bin/bash"
    my_rc_params["shebang"] = f"#!{shebang}"
    run_commands(
        ['command="$(ps -p $$ -o cmd | tail -n 1)"', f'[ "${{command% *}}" = {shebang} ]'],
        rc_params=my_rc_params,
    )


def test_renders_profile_template(my_rc_params):
    my_rc_params["my_command"] = "false"
    my_rc_params["profile_template_content"] = "{{{my_command}}}"
    with raises(RuntimeError):
        run_commands("true", rc_params=my_rc_params)


def test_uses_profile_template_path(my_rc_params):
    my_rc_params["my_command"] = "false"
    with NamedTemporaryFile("w") as file:
        file.write("{{{my_command}}}")
        file.flush()
        my_rc_params["profile_template_path"] = file.name
        with raises(RuntimeError):
            run_commands("true", rc_params=my_rc_params)


def test_respects_picongpurc_path(my_rc_params):
    with NamedTemporaryFile("wb") as file:
        tomli_w.dump({"profile_content": '"false"'}, file)
        file.flush()
        my_rc_params["picongpurc_path"] = file.name
        with raises(RuntimeError):
            run_commands("true", rc_params=my_rc_params)


def test_find_picongpurc_same_directory():
    content = "Hello World!"
    filename = "tmp-custom-name"
    with TemporaryDirectory() as d1:
        with (Path(d1) / filename).open("w") as file:
            file.write(content)
        with find_picongpurc_path(search_file=filename, search_path=d1).open("rb") as file:
            assert file.read().decode() == content


def test_find_picongpurc_parent_directory():
    content = "Hello World!"
    filename = "tmp-custom-name"
    with TemporaryDirectory() as d1:
        with (Path(d1) / filename).open("w") as file:
            file.write(content)
        with TemporaryDirectory(prefix=f"{d1}/") as d2:
            with find_picongpurc_path(search_file=filename, search_path=d2).open("rb") as file:
                assert file.read().decode() == content
