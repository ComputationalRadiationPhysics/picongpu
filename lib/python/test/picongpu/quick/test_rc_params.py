# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from moosetash import MissingVariable
from picongpu import DirtyResetError, core, rc_params
from picongpu._rc_params import search_for_in_parents
from pytest import fixture, raises, warns


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
def arbitrary_filename():
    return "tmp-custom-filename"


@fixture
def dirty_rc_params(my_rc_params, any_key, any_content):
    my_rc_params["dirty_reset_policy"] = "ignore"
    my_rc_params[any_key] = any_content
    assert my_rc_params[any_key] == any_content
    return my_rc_params


def test_presets_set_src_path(my_rc_params):
    my_rc_params["preset"] = "bash/bash_picongpu"
    assert my_rc_params["pic_src_path"] == core.path()


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
    previous = deepcopy(dirty_rc_params)._data
    dirty_rc_params["preset"] = "bash/bash_picongpu"

    assert call_args == ["preset", "bash/bash_picongpu", previous, dirty_rc_params._data]


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


def test_set_temporarily(my_rc_params, any_key, any_content):
    assert any_key not in my_rc_params
    with my_rc_params.set_temporarily(any_key=any_content):
        assert my_rc_params["any_key"] == any_content
    assert any_key not in my_rc_params


def test_set_temporarily_overrides_existing_by_default(my_rc_params, any_key, any_content):
    other_content = "other cool content"
    assert other_content != any_content

    my_rc_params[any_key] = other_content
    assert my_rc_params["any_key"] == other_content
    with my_rc_params.set_temporarily(any_key=any_content):
        assert my_rc_params["any_key"] == any_content
    assert my_rc_params["any_key"] == other_content


def test_set_temporarily_allows_not_overriding_existing(my_rc_params, any_key, any_content):
    other_content = "other cool content"
    assert other_content != any_content

    my_rc_params[any_key] = other_content
    assert my_rc_params["any_key"] == other_content
    with my_rc_params.set_temporarily(any_key=any_content, override_existing=False):
        assert my_rc_params["any_key"] == other_content
    assert my_rc_params["any_key"] == other_content


def test_search_for_file_in_same_directory(any_content, arbitrary_filename):
    with TemporaryDirectory() as d1:
        with (Path(d1) / arbitrary_filename).open("w") as file:
            file.write(any_content)
        with search_for_in_parents(filename=arbitrary_filename, start_path=d1).open("rb") as file:
            assert file.read().decode() == any_content


def test_search_for_file_in_parent_directory(any_content, arbitrary_filename):
    with TemporaryDirectory() as d1:
        with (Path(d1) / arbitrary_filename).open("w") as file:
            file.write(any_content)
        with TemporaryDirectory(prefix=f"{d1}/") as d2:
            with search_for_in_parents(filename=arbitrary_filename, start_path=d2).open("rb") as file:
                assert file.read().decode() == any_content


def test_search_for_file_returns_none_if_not_found(arbitrary_filename):
    with TemporaryDirectory() as d1:
        assert search_for_in_parents(filename=arbitrary_filename, start_path=d1) is None
