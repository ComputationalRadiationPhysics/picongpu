"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import re
import subprocess
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from moosetash import MissingVariable
from picongpu import DirtyResetError, _rc_params, core, rc_params
from picongpu._rc_params import (
    PRESET_STORAGE_PATH,
    PROFILE_PARAMETERS,
    generate_default_rc_params,
    get_preset_declared_value,
    get_preset_name,
    get_preset_path,
    get_profile_parameter,
    preset_default_prefill,
    search_for_in_parents,
    search_in_user_config,
    user_config_path,
)
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
    my_rc_params["pic_libs"] = "$HOME/pic-libs"
    assert f'export PICSRC="{str(my_rc_params["pic_src_path"])}"' in my_rc_params.profile_content


def test_knows_about_its_required_information(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    # We don't want to test exhaustively here because things might change.
    # Just using one sample that is likely to be retained.
    assert "author" in my_rc_params["required_information"]


def test_profile_parameters_have_unique_keys():
    rc_keys = [parameter.rc_key for parameter in PROFILE_PARAMETERS]
    assert len(rc_keys) == len(set(rc_keys))


def test_profile_parameters_have_descriptions():
    for parameter in PROFILE_PARAMETERS:
        assert parameter.description, parameter.rc_key


def test_the_exact_required_parameter_set_is_pinned():
    # This is the regression guard for the is_required flags: flipping any of
    # them (e.g. pic_libs -> False) must not go unnoticed.
    assert {parameter.rc_key for parameter in PROFILE_PARAMETERS if parameter.is_required} == {
        "author",
        "email",
        "pic_libs",
        "pic_src_path",
        "project_id",
        "project_name",
        "scratch_dir",
    }


def test_get_profile_parameter_roundtrips():
    for parameter in PROFILE_PARAMETERS:
        assert get_profile_parameter(parameter.rc_key) is parameter
    assert get_profile_parameter("no_such_parameter") is None


def test_preset_default_prefill_for_rosi_pic_libs():
    # rosi-hzdr pins pic_libs to a concrete, site-provided directory that the
    # user can just accept (https://github.com/chillenzer-agents/picongpu/issues/204).
    assert preset_default_prefill("rosi-hzdr/gpu-v100", "pic_libs") == "/bigdata/hplsim/development/rosi-picongpu-libs/"


def test_preset_default_prefill_ignores_other_parameters_and_presets():
    # Only the explicitly registered (preset, key) pairs pre-fill.
    assert preset_default_prefill("rosi-hzdr/gpu-v100", "author") is None
    # jupiter/perlmutter declare a *shell* default ("$PROJECT/...") that the
    # user must expand themselves; it must not be offered as a literal prefill.
    assert preset_default_prefill("jupiter-jsc/gh200", "pic_libs") is None
    assert preset_default_prefill("perlmutter-nersc/gpu", "pic_libs") is None
    assert preset_default_prefill(None, "pic_libs") is None


def test_preset_default_prefill_is_gated_on_registration_not_literalness():
    # capella declares a literal (placeholder-*looking*, but non-shell) default
    # for project_name; it must NOT pre-fill because the (preset, key) pair is
    # not registered. This pins the allowlist gate itself: dropping the
    # membership test would make this pass through as "p_name_of_your_project".
    assert get_preset_declared_value("zih-tud/capella", "project_name") == "p_name_of_your_project"
    assert preset_default_prefill("zih-tud/capella", "project_name") is None


def test_get_preset_declared_value_returns_none_for_unknown_preset():
    # An unknown preset makes _read_preset raise ValueError, which the helper
    # turns into None rather than propagating.
    assert get_preset_declared_value("does-not-exist/nope", "pic_libs") is None


def test_get_preset_declared_value_only_returns_concrete_literals():
    assert (
        get_preset_declared_value("rosi-hzdr/gpu-v100", "pic_libs") == "/bigdata/hplsim/development/rosi-picongpu-libs/"
    )
    # shell expansion / command substitution / placeholders are not literal:
    assert get_preset_declared_value("jupiter-jsc/gh200", "pic_libs") is None
    assert get_preset_declared_value("frontier-ornl/batch", "project_id") is None
    assert get_preset_declared_value("rosi-hzdr/gpu-v100", "no_such_key") is None


def test_required_information_is_exactly_the_required_preset_parameters(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    required = set(my_rc_params["required_information"])
    # only parameters the preset actually declares can be part of it ...
    assert required <= {p.rc_key for p in PROFILE_PARAMETERS if p.is_required}
    # ... and every required parameter the preset declares must be in it.
    assert {"author", "email", "pic_src_path"} <= required


def test_non_required_parameters_are_kept_as_default(my_rc_params):
    my_rc_params["preset"] = "hemera-hzdr/defq_picongpu"
    assert "tbg_submit" not in my_rc_params["required_information"]
    assert "tbg_submit" in my_rc_params


def test_required_partition_for_a_preset_declaring_optional_keys(my_rc_params):
    # perlmutter declares pic_libs/project_id, so it exercises the required
    # flags that hemera-hzdr/defq_picongpu never touches.
    my_rc_params["preset"] = "perlmutter-nersc/gpu"
    assert set(my_rc_params["required_information"]) == {
        "author",
        "email",
        "pic_src_path",
        "pic_libs",
        "project_id",
    }
    # a non-required parameter declared by this preset stays a retained default
    assert "tbg_submit" not in my_rc_params["required_information"]
    assert "tbg_submit" in my_rc_params


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


_PROFILE_ENV_VARS = (
    "PIC_LIBS",
    "CMAKE_ROOT",
    "BOOST_ROOT",
    "BLOSC_ROOT",
    "LIBPNG_ROOT",
    "PNGwriter_ROOT",
    "HDF5_ROOT",
    "ADIOS2_ROOT",
    "OPENPMD_ROOT",
    "FFTW_ROOT",
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "LD_LIBRARY_PATH",
)


def _source_bash_profile(home, **env):
    profile = core.path("etc") / "picongpu" / "bash" / "bash_picongpu.profile.example"
    body = "\n".join(f'printf "%s=%s\\n" {v} "${{{v}-<UNSET>}}"' for v in _PROFILE_ENV_VARS)
    proc = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", f'source "$1" >/dev/null 2>&1\n{body}', "_", profile],
        env={"HOME": str(home), "PATH": "/usr/local/bin:/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


def test_bash_profile_leaves_a_clean_environment_untouched(tmp_path):
    # Backwards compatibility: a user relying on system/distro packages neither
    # gets PIC_LIBS forced nor any *_ROOT / search-path entries injected.
    values = _source_bash_profile(tmp_path)
    assert values == {v: "<UNSET>" for v in _PROFILE_ENV_VARS}


def test_bash_profile_preserves_pre_set_roots_and_search_paths(tmp_path):
    # A user who already configured a self-build (or any prefix) keeps their
    # values; the profile only fills in what is unset.
    values = _source_bash_profile(
        tmp_path,
        PIC_LIBS="/opt/custom-piclibs",
        CMAKE_ROOT="/opt/cmake",
        BOOST_ROOT="/opt/boost",
        CMAKE_PREFIX_PATH="/opt/pfx",
        CPATH="/opt/inc",
        LD_LIBRARY_PATH="/opt/lib",
    )
    assert values["PIC_LIBS"] == "/opt/custom-piclibs"
    assert values["CMAKE_ROOT"] == "/opt/cmake"
    assert values["BOOST_ROOT"] == "/opt/boost"
    assert values["CMAKE_PREFIX_PATH"] == "/opt/pfx"
    assert values["CPATH"] == "/opt/inc"
    assert values["LD_LIBRARY_PATH"] == "/opt/lib"


def test_bash_profile_is_reproduced(my_rc_params):
    my_rc_params["preset"] = "bash/bash_picongpu"
    # The shipped example keeps PIC_LIBS unset unless a prefix is provided;
    # rendering the same expression must reproduce it byte-for-byte.
    my_rc_params["pic_libs"] = "${PIC_LIBS:-}"
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


def test_bare_bash_resolves_canonically_with_variants_present():
    # Adding bash/spack and bash/conda must not make preset="bash" ambiguous.
    assert get_preset_path("bash") == PRESET_STORAGE_PATH / "bash" / "bash_picongpu.profile.example"
    assert get_preset_path("bash/bash_picongpu.profile.example") == get_preset_path("bash")


def test_bash_variant_presets_resolve_their_own_profiles():
    assert get_preset_path("bash/spack") == PRESET_STORAGE_PATH / "bash" / "spack_picongpu.profile.example"
    assert get_preset_path("bash/conda") == PRESET_STORAGE_PATH / "bash" / "conda_picongpu.profile.example"
    assert get_preset_path("bash/spack_picongpu.profile.example") == get_preset_path("bash/spack")


def test_preset_names_are_canonical():
    assert get_preset_name("bash/bash_picongpu.profile.example") == "bash"
    assert get_preset_name("bash/spack_picongpu.profile.example") == "bash/spack"
    assert get_preset_name("bash/conda_picongpu.profile.example") == "bash/conda"
    assert get_preset_name("hemera-hzdr/gpu_picongpu.profile.example") == "hemera-hzdr/gpu"
    assert get_preset_name("perlmutter-nersc/gpu.profile.example") == "perlmutter-nersc/gpu"


def _exported_version_pins(profile_name):
    text = (PRESET_STORAGE_PATH / "bash" / profile_name).read_text()
    return dict(re.findall(r'export ([A-Z][A-Z0-9_]*_VERSION)="\$\{[A-Z0-9_]+:-([^}]*)\}"', text))


def test_bash_profiles_share_version_pins_and_environment():
    profiles = [
        "bash_picongpu.profile.example",
        "spack_picongpu.profile.example",
        "conda_picongpu.profile.example",
    ]
    pins = [_exported_version_pins(name) for name in profiles]
    assert pins[0], "failed to parse the version pins from the base profile"
    assert pins[0] == pins[1] == pins[2]

    texts = [(PRESET_STORAGE_PATH / "bash" / name).read_text() for name in profiles]
    for line in (
        'export PIC_SYSTEM_TEMPLATE_PATH=${PIC_SYSTEM_TEMPLATE_PATH:-"etc/picongpu/bash"}',
        'export PIC_BACKEND="omp2b:native"',
    ):
        assert all(line in text for text in texts), line


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


def test_user_config_path_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert user_config_path() == tmp_path / "xdg" / "picongpu" / "picongpurc.toml"


def test_user_config_path_falls_back_to_home_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert user_config_path() == tmp_path / "home" / ".config" / "picongpu" / "picongpurc.toml"


def test_user_config_path_treats_empty_xdg_as_unset(monkeypatch, tmp_path):
    # An empty XDG_CONFIG_HOME must behave like an unset one; reverting the
    # `or` fallback to environ.get(..., default) would make this fail.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert user_config_path() == tmp_path / "home" / ".config" / "picongpu" / "picongpurc.toml"


def test_user_config_path_ignores_relative_xdg_config_home(monkeypatch, tmp_path):
    # The XDG spec requires absolute paths; a relative value is invalid and
    # must be ignored in favour of ~/.config.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")
    assert user_config_path() == tmp_path / "home" / ".config" / "picongpu" / "picongpurc.toml"


def test_user_config_path_expands_a_tilde_in_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", "~/.config")
    assert user_config_path() == tmp_path / "home" / ".config" / "picongpu" / "picongpurc.toml"


def test_search_in_user_config_finds_the_xdg_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    target = tmp_path / "xdg" / "picongpu" / "picongpurc.toml"
    target.parent.mkdir(parents=True)
    target.write_text('preset = "bash"\n')
    assert search_in_user_config() == target


def test_discovery_precedence_pic_rc_then_dot_name_then_xdg(monkeypatch, tmp_path):
    # Keep the documented precedence: PIC_RC, then the nearest dot-prefixed
    # file up the tree, then the XDG user configuration
    # (https://github.com/chillenzer-agents/picongpu/issues/206).
    home = tmp_path / "home"
    xdg_file = home / ".config" / "picongpu" / "picongpurc.toml"
    xdg_file.parent.mkdir(parents=True)
    xdg_file.write_text('preset = "bash"\n')

    project = tmp_path / "project"
    run_dir = project / "run"
    run_dir.mkdir(parents=True)
    dot_file = project / ".picongpurc.toml"
    dot_file.write_text('preset = "bash"\n')

    override = tmp_path / "override.toml"
    override.write_text('preset = "bash"\n')

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PIC_RC", raising=False)
    monkeypatch.chdir(run_dir)

    # A dot-name closer to the input wins over the XDG location.
    assert generate_default_rc_params()["picongpurc_path"] == dot_file

    # PIC_RC wins over everything.
    monkeypatch.setenv("PIC_RC", str(override))
    assert generate_default_rc_params()["picongpurc_path"] == override

    # Without a closer dot-name or PIC_RC, the XDG file is used.
    # Stub the upward search here: it reaches the filesystem root, so a
    # machine-level /.picongpurc.toml (e.g. written by the CI setup) would
    # otherwise shadow the XDG file. The search itself is covered by the
    # dedicated search_for_in_parents tests above.
    monkeypatch.delenv("PIC_RC")
    dot_file.unlink()
    monkeypatch.setattr(_rc_params, "search_for_in_parents", lambda *args, **kwargs: None)
    assert generate_default_rc_params()["picongpurc_path"] == xdg_file
