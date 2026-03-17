"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from copy import deepcopy

from moosetash import MissingVariable
from picongpu import rc_params, DirtyResetError
from pytest import fixture, raises, warns

from picongpu import core


@fixture
def my_rc_params():
    return type(rc_params)()


@fixture
def any_key():
    return "any_key"


@fixture
def any_content():
    return "any content"


@fixture
def dirty_rc_params(my_rc_params, any_key, any_content):
    my_rc_params["dirty_reset_policy"] = "ignore"
    my_rc_params[any_key] = any_content
    assert my_rc_params[any_key] == any_content
    return my_rc_params


def test_presets_set_src_path(my_rc_params):
    my_rc_params["preset"] = "bash/bash_picongpu"
    assert "pic_src_path" in my_rc_params


def test_presets_clear_previous_settings(dirty_rc_params, any_key):
    dirty_rc_params["preset"] = "bash/bash_picongpu"
    assert any_key not in dirty_rc_params


def test_dirty_resets_can_raise(dirty_rc_params):
    dirty_rc_params["dirty_reset_policy"] = "raise"
    with raises(DirtyResetError):
        dirty_rc_params["preset"] = "bash/bash_picongpu"


def test_dirty_resets_can_warn(dirty_rc_params):
    dirty_rc_params["dirty_reset_policy"] = "warn"
    with warns():
        dirty_rc_params["preset"] = "bash/bash_picongpu"


def test_dirty_resets_can_ignore(dirty_rc_params):
    dirty_rc_params["dirty_reset_policy"] = "ignore"
    # no warning or exception:
    dirty_rc_params["preset"] = "bash/bash_picongpu"


def test_dirty_resets_can_trigger_custom_handler(dirty_rc_params):
    call_args = []
    dirty_rc_params["dirty_reset_policy"] = lambda *args: call_args.extend(args) or args[3]
    previous = deepcopy(dirty_rc_params)
    dirty_rc_params["preset"] = "bash/bash_picongpu"

    assert call_args == ["preset", "bash/bash_picongpu", previous, dirty_rc_params]


def test_preset_points_profile_to_correct_pic_src_path(my_rc_params):
    my_rc_params["preset"] = "bash/bash_picongpu"
    assert f'export PICSRC="{str(my_rc_params["pic_src_path"])}"' in my_rc_params.profile_content


def test_knows_about_its_required_information(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    # We don't want to test exhaustively here because things might change.
    # Just using one sample that is likely to be retained.
    assert "author" in my_rc_params["required_information"]


def test_rendering_profile_with_incomplete_info_raises(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    with raises(MissingVariable):
        my_rc_params.profile_content


def test_rendering_profile_with_incomplete_info_can_warn(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    my_rc_params["missing_variable_policy"] = "warn"
    with warns():
        my_rc_params.profile_content


def test_rendering_profile_with_incomplete_info_can_ignore(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    my_rc_params["missing_variable_policy"] = "ignore"
    # passes:
    my_rc_params.profile_content


def test_rendering_profile_with_incomplete_info_can_trigger_custom_handler(my_rc_params):
    call_args = []
    my_rc_params["missing_variable_policy"] = lambda *args: call_args.extend(args) or ""
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"

    my_rc_params.profile_content

    assert "author" in call_args


def test_bash_profile_is_reproduced(my_rc_params):
    my_rc_params["preset"] = "bash/bash_picongpu"
    my_rc_params["pic_src_path"] = "$HOME/src/picongpu"
    with (core.path("etc") / "picongpu" / "bash" / "bash_picongpu.profile.example").open("r") as file:
        assert my_rc_params.profile_content == file.read()


def _no_comment_or_blank_line(line):
    return not line.startswith("#") and not line.strip() == ""


def test_hemera_profile_is_reproduced(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    # The dummy variable content used in the example:
    my_rc_params |= {
        "pic_src_path": "$HOME/src/picongpu",
        "email": "someone@example.com",
        "author": "$(whoami) <$MY_MAIL>",
    }
    profile_content = "\n".join(filter(_no_comment_or_blank_line, my_rc_params.profile_content.split("\n")))
    with (core.path("etc") / "picongpu" / "hemera-hzdr" / "defq_picongpu.profile.example").open("r") as file:
        expected = "".join(filter(_no_comment_or_blank_line, file.readlines())).strip()
    assert profile_content == expected


def test_raises_on_non_existent_preset(my_rc_params):
    with raises(ValueError):
        my_rc_params["preset"] = "bogus"


def test_raises_on_ambiguous_preset(my_rc_params):
    with raises(ValueError):
        my_rc_params["preset"] = "hemera-hzdr"


def test_allows_short_name_for_unambiguous_presets(my_rc_params):
    # passes:
    my_rc_params["preset"] = "bash"


def test_allows_file_extension_for_presets(my_rc_params):
    # passes:
    my_rc_params["preset"] = "bash/bash_picongpu.profile.example"
