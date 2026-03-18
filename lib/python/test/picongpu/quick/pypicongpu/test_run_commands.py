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
from picongpu._rc_params import RCParams, search_for_in_parents
from picongpu.pypicongpu.runner import generate_bare_profile_as_in, run_commands, generate_bare_profile
from pytest import fixture, mark, raises


@fixture
def my_rc_params():
    return type(rc_params)()


@fixture
def arbitrary_string():
    return "Hello World"


@fixture
def arbitrary_filename():
    return "tmp-custom-filename"


@fixture
def failing_command():
    return "false"


@fixture
def passing_command():
    return "true"


def test_raises_on_failing_command(failing_command):
    with raises(RuntimeError):
        run_commands(failing_command)


def test_raises_on_failing_command_with_others_later(failing_command, passing_command):
    with raises(RuntimeError):
        run_commands([failing_command, passing_command])


def test_passes_failures_with_overriden_preamble(my_rc_params, failing_command, passing_command):
    my_rc_params["preamble"] = ""
    # Does not raise a RuntimeError as above
    # because we have overridden the preamble
    # that applied `set -e` among other things.
    run_commands([failing_command, passing_command], rc_params=my_rc_params)


def test_uses_profile_content(my_rc_params, arbitrary_string, passing_command):
    my_rc_params["profile_content"] = f'echo "{arbitrary_string}"'
    assert run_commands(passing_command, rc_params=my_rc_params).stdout.decode().strip() == arbitrary_string


def test_uses_profile_path(my_rc_params, passing_command, arbitrary_string):
    with NamedTemporaryFile("w") as file:
        file.write(f'echo "{arbitrary_string}"')
        file.flush()
        my_rc_params["profile_path"] = file.name
        assert run_commands(passing_command, rc_params=my_rc_params).stdout.decode().strip() == arbitrary_string


# Taking a little guess here that these might be available on test systems:
@mark.parametrize("shell", filter(lambda p: Path(p).exists(), ("/bin/sh", "/bin/bash", "/bin/zsh")))
def test_respects_shebang_in_profile(my_rc_params, shell):
    my_rc_params["profile_content"] = f"#!{shell}"
    run_commands(
        # This is a test for the executable that's running.
        # The process would fail and raise a RuntimeError.
        # So, no assertion here, succeeding command is enough.
        ['command="$(ps -p $$ -o cmd | tail -n 1)"', f'[ "${{command% *}}" = {shell} ]'],
        rc_params=my_rc_params,
    )


@mark.skipif(
    condition=(not Path("/bin/sh").exists()) or (not Path("/bin/bash").exists()),
    reason="The used shells are not available.",
)
def test_respects_overriden_shebang(my_rc_params):
    shebang = "/bin/sh"
    my_rc_params["profile_content"] = "#!/bin/bash"
    my_rc_params["shebang"] = f"#!{shebang}"
    run_commands(
        ['command="$(ps -p $$ -o cmd | tail -n 1)"', f'[ "${{command% *}}" = {shebang} ]'],
        rc_params=my_rc_params,
    )


def test_renders_profile_template(my_rc_params, arbitrary_string, passing_command):
    my_rc_params["my_command"] = f'echo "{arbitrary_string}"'
    my_rc_params["profile_template_content"] = "{{{my_command}}}"
    assert run_commands(passing_command, rc_params=my_rc_params).stdout.decode().strip() == arbitrary_string


def test_uses_profile_template_path(my_rc_params, arbitrary_string, passing_command):
    my_rc_params["my_command"] = f'echo "{arbitrary_string}"'
    with NamedTemporaryFile("w") as file:
        file.write("{{{my_command}}}")
        file.flush()
        my_rc_params["profile_template_path"] = file.name
        assert run_commands(passing_command, rc_params=my_rc_params).stdout.decode().strip() == arbitrary_string


def test_respects_picongpurc_path(my_rc_params, arbitrary_string, passing_command):
    with NamedTemporaryFile("wb") as file:
        tomli_w.dump({"profile_content": f'echo "{arbitrary_string}"'}, file)
        file.flush()
        my_rc_params["picongpurc_path"] = file.name
        assert run_commands(passing_command, rc_params=my_rc_params).stdout.decode().strip() == arbitrary_string


def test_search_for_file_in_same_directory(arbitrary_string, arbitrary_filename):
    with TemporaryDirectory() as d1:
        with (Path(d1) / arbitrary_filename).open("w") as file:
            file.write(arbitrary_string)
        with search_for_in_parents(filename=arbitrary_filename, start_path=d1).open("rb") as file:
            assert file.read().decode() == arbitrary_string


def test_search_for_file_in_parent_directory(arbitrary_string, arbitrary_filename):
    with TemporaryDirectory() as d1:
        with (Path(d1) / arbitrary_filename).open("w") as file:
            file.write(arbitrary_string)
        with TemporaryDirectory(prefix=f"{d1}/") as d2:
            with search_for_in_parents(filename=arbitrary_filename, start_path=d2).open("rb") as file:
                assert file.read().decode() == arbitrary_string


def test_search_for_file_returns_none_if_not_found(arbitrary_filename):
    with TemporaryDirectory() as d1:
        assert search_for_in_parents(filename=arbitrary_filename, start_path=d1) is None


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
