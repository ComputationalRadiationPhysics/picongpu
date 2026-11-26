"""
Tests for pic-deps (lib/python/picongpu/pic_deps.py).
"""

import os
from pathlib import Path

import pytest

from picongpu._rc_params import PRESET_STORAGE_PATH, RCParams
from picongpu.pic_deps import (
    CUSTOM_SCRIPT_KEY,
    check_dependencies,
    install_dependencies,
    parse_guard_roots,
    resolve_preset_script,
)

_AMBIENT_DEPENDENCY_VARS = (
    "PIC_LIBS",
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "LD_LIBRARY_PATH",
    "CONDA_PREFIX",
    "OPAL_PREFIX",
)


@pytest.fixture
def clean_dependency_environment(monkeypatch):
    """Hide the ambient toolchain environment from the profile-sourcing tests.

    CI exports pinned ``*_VERSION`` variables (e.g. ``BOOST_VERSION=1.75.0``)
    and agc-manager ``*_ROOT`` variables whose ``/opt/...`` trees exist, which
    would otherwise override the defaults asserted here.
    """
    for name in list(os.environ):
        if name.endswith(("_VERSION", "_ROOT")) or name in _AMBIENT_DEPENDENCY_VARS:
            monkeypatch.delenv(name, raising=False)


def _perlmutter_script() -> Path:
    return PRESET_STORAGE_PATH / "perlmutter-nersc" / "dependencies_autoinstall.sh"


def _completed_params():
    p = RCParams(preset="perlmutter-nersc/gpu.profile.example")
    p["author"] = "me"
    p["email"] = "me@example.com"
    p["pic_libs"] = str(Path(__file__).parent / "nonexistent-piclibs")
    p["pic_src_path"] = str(Path(__file__).parent / "nonexistent-picsrc")
    p["project_id"] = "m0000"
    return p


def test_parse_guard_roots_perlmutter():
    assert parse_guard_roots(_perlmutter_script()) == [
        "BOOST_ROOT",
        "BLOSC_ROOT",
        "PNGwriter_ROOT",
        "ADIOS2_ROOT",
        "OPENPMD_ROOT",
    ]


def test_parse_guard_roots_delta_includes_fftw_and_libpng():
    roots = parse_guard_roots(PRESET_STORAGE_PATH / "delta-ncsa" / "dependencies_autoinstall.sh")
    assert "FFTW_ROOT" in roots
    assert "LIBPNG_ROOT" in roots
    assert "HDF5_ROOT" in roots
    # de-duplicated, ordered
    assert len(roots) == len(set(roots))


def test_resolve_preset_script_ok():
    preset_dir, script = resolve_preset_script(_completed_params())
    assert preset_dir == "perlmutter-nersc"
    assert script.is_file()
    assert script == _perlmutter_script()


def test_resolve_preset_script_no_preset():
    p = RCParams()
    p["preset"] = None
    with pytest.raises(SystemExit, match="no preset"):
        resolve_preset_script(p)


def _completed_bash_params():
    p = RCParams(preset="bash/bash_picongpu.profile.example")
    p["author"] = "me"
    p["email"] = "me@example.com"
    p["pic_libs"] = str(Path(__file__).parent / "nonexistent-piclibs")
    p["pic_src_path"] = str(Path(__file__).parent / "nonexistent-picsrc")
    return p


def test_resolve_preset_script_bash_is_supported():
    preset_dir, script = resolve_preset_script(_completed_bash_params())
    assert preset_dir == "bash"
    assert script.is_file()
    assert script == PRESET_STORAGE_PATH / "bash" / "dependencies_autoinstall.sh"


def test_parse_guard_roots_bash():
    assert parse_guard_roots(PRESET_STORAGE_PATH / "bash" / "dependencies_autoinstall.sh") == [
        "BOOST_ROOT",
        "BLOSC_ROOT",
        "LIBPNG_ROOT",
        "PNGwriter_ROOT",
        "HDF5_ROOT",
        "ADIOS2_ROOT",
        "OPENPMD_ROOT",
        "FFTW_ROOT",
    ]


def test_resolve_preset_script_missing_script_for_module_preset():
    # presets that rely on system modules still have no autoinstall script
    p = RCParams(preset="hemera-hzdr/defq_picongpu.profile.example")
    with pytest.raises(SystemExit, match="dependencies_autoinstall.sh"):
        resolve_preset_script(p)


def test_check_dependencies_reports_all_missing_for_empty_piclibs(clean_dependency_environment):
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    results = check_dependencies(_completed_params())
    roots = [r for r, _, _ in results]
    assert roots == ["BOOST_ROOT", "BLOSC_ROOT", "PNGwriter_ROOT", "ADIOS2_ROOT", "OPENPMD_ROOT"]
    # pic_libs points at a non-existent dir -> every dep is missing
    assert all(ok is False for _, _, ok in results)
    # resolved paths are the profile-derived versioned roots (non-empty)
    assert all(resolved for _, resolved, _ in results)


def test_check_dependencies_bash_enumerates_profile_roots(clean_dependency_environment):
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    results = check_dependencies(_completed_bash_params())
    roots = [r for r, _, _ in results]
    assert roots == [
        "BOOST_ROOT",
        "BLOSC_ROOT",
        "LIBPNG_ROOT",
        "PNGwriter_ROOT",
        "HDF5_ROOT",
        "ADIOS2_ROOT",
        "OPENPMD_ROOT",
        "FFTW_ROOT",
    ]
    # pic_libs points at a non-existent dir -> every dep is missing
    assert all(ok is False for _, _, ok in results)
    # the bash profile derives each *_ROOT from PIC_LIBS
    assert all(resolved.startswith(str(Path(__file__).parent / "nonexistent-piclibs")) for _, resolved, _ in results)


def _completed_bash_variant_params(preset):
    p = RCParams(preset=preset)
    p["author"] = "me"
    p["email"] = "me@example.com"
    p["pic_libs"] = str(Path(__file__).parent / "nonexistent-piclibs")
    p["pic_src_path"] = str(Path(__file__).parent / "nonexistent-picsrc")
    return p


def test_resolve_preset_script_bash_spack_selects_variant_script():
    preset_dir, script = resolve_preset_script(_completed_bash_variant_params("bash/spack"))
    assert preset_dir == "bash"
    assert script == PRESET_STORAGE_PATH / "bash" / "dependencies_autoinstall_spack.sh"


def test_resolve_preset_script_bash_conda_selects_variant_script():
    preset_dir, script = resolve_preset_script(_completed_bash_variant_params("bash/conda"))
    assert preset_dir == "bash"
    assert script == PRESET_STORAGE_PATH / "bash" / "dependencies_autoinstall_conda.sh"


def test_parse_guard_roots_spack_and_conda_keep_the_full_root_contract():
    expected = [
        "BOOST_ROOT",
        "BLOSC_ROOT",
        "LIBPNG_ROOT",
        "PNGwriter_ROOT",
        "HDF5_ROOT",
        "ADIOS2_ROOT",
        "OPENPMD_ROOT",
        "FFTW_ROOT",
    ]
    assert parse_guard_roots(PRESET_STORAGE_PATH / "bash" / "dependencies_autoinstall_spack.sh") == expected
    assert parse_guard_roots(PRESET_STORAGE_PATH / "bash" / "dependencies_autoinstall_conda.sh") == expected


def test_resolve_preset_script_error_lists_available_scripts():
    p = RCParams(preset="hemera-hzdr/defq_picongpu.profile.example")
    with pytest.raises(SystemExit, match=r"Available scripts: \(none\)"):
        resolve_preset_script(p)


def test_check_dependencies_conda_enumerates_shared_prefix_roots(clean_dependency_environment):
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    pic_libs = Path(__file__).parent / "nonexistent-piclibs"
    results = check_dependencies(_completed_bash_variant_params("bash/conda"))
    roots = [r for r, _, _ in results]
    assert roots == [
        "BOOST_ROOT",
        "BLOSC_ROOT",
        "LIBPNG_ROOT",
        "PNGwriter_ROOT",
        "HDF5_ROOT",
        "ADIOS2_ROOT",
        "OPENPMD_ROOT",
        "FFTW_ROOT",
    ]
    assert all(ok is False for _, _, ok in results)
    resolved = dict((root, path) for root, path, _ in results)
    for root in ("BOOST_ROOT", "BLOSC_ROOT", "LIBPNG_ROOT", "HDF5_ROOT", "ADIOS2_ROOT", "OPENPMD_ROOT", "FFTW_ROOT"):
        assert resolved[root] == str(pic_libs / "conda")
    assert resolved["PNGwriter_ROOT"] == str(pic_libs / "pngwriter" / "0.7.0")


def test_check_dependencies_spack_enumerates_versioned_roots(clean_dependency_environment):
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    pic_libs = Path(__file__).parent / "nonexistent-piclibs"
    results = check_dependencies(_completed_bash_variant_params("bash/spack"))
    resolved = dict((root, path) for root, path, _ in results)
    assert resolved["BOOST_ROOT"] == str(pic_libs / "boost" / "1.87.0")
    assert resolved["HDF5_ROOT"] == str(pic_libs / "hdf5" / "2.0.0")
    assert resolved["PNGwriter_ROOT"] == str(pic_libs / "pngwriter" / "0.7.0")


# --- user-supplied dependency installer scripts (pic_deps_script / --script) ---

_CUSTOM_SCRIPT = """#!/usr/bin/env bash
# harmless custom installer used by the tests
source "$PIC_PROFILE"
if [ ! -d "$MYTEST_ROOT" ]; then
    mkdir -p "$MYTEST_ROOT"
fi
"""


def _write_custom_script(tmp_path, name="my_dependencies_autoinstall.sh"):
    script = tmp_path / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_CUSTOM_SCRIPT)
    return script


def test_resolve_preset_script_custom_rc_parameter(tmp_path):
    script = _write_custom_script(tmp_path)
    p = _completed_params()
    p[CUSTOM_SCRIPT_KEY] = str(script)
    preset_dir, resolved = resolve_preset_script(p)
    assert preset_dir == "perlmutter-nersc"
    assert resolved == script.absolute()


def test_resolve_preset_script_custom_without_preset(tmp_path):
    script = _write_custom_script(tmp_path)
    p = RCParams()
    p[CUSTOM_SCRIPT_KEY] = str(script)
    preset_dir, resolved = resolve_preset_script(p)
    assert preset_dir == "custom"
    assert resolved == script.absolute()


def test_resolve_preset_script_custom_cli_overrides_rc_parameter(tmp_path):
    rc_script = _write_custom_script(tmp_path / "rc", name="rc.sh")
    cli_script = _write_custom_script(tmp_path / "cli", name="cli.sh")
    p = _completed_params()
    p[CUSTOM_SCRIPT_KEY] = str(rc_script)
    _, resolved = resolve_preset_script(p, script=str(cli_script))
    assert resolved == cli_script.absolute()


def test_resolve_preset_script_custom_missing_script(tmp_path):
    p = _completed_params()
    p[CUSTOM_SCRIPT_KEY] = str(tmp_path / "does-not-exist.sh")
    with pytest.raises(SystemExit, match="does not exist|not a file"):
        resolve_preset_script(p)


def test_check_and_install_with_configured_script(tmp_path, clean_dependency_environment):
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    root = tmp_path / "root"
    script = _write_custom_script(tmp_path)
    p = RCParams()
    p[CUSTOM_SCRIPT_KEY] = str(script)
    p["profile_content"] = f'export MYTEST_ROOT="{root}"\n'

    # before install: the guard directory does not exist yet
    assert check_dependencies(p) == [("MYTEST_ROOT", str(root), False)]

    # the configured script is run (it creates the guard directory) and the
    # final check succeeds
    assert install_dependencies(p) == 0
    assert root.is_dir()
    assert check_dependencies(p) == [("MYTEST_ROOT", str(root), True)]


def test_main_with_custom_script_without_rc_key(tmp_path, monkeypatch, clean_dependency_environment):
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    import picongpu.pic_deps as pic_deps

    script = _write_custom_script(tmp_path)
    root = tmp_path / "root"
    p = RCParams()
    p["profile_content"] = f'export MYTEST_ROOT="{root}"\n'
    monkeypatch.setattr(pic_deps, "_rc_params_default", p)

    assert pic_deps.main(["check", "--script", str(script)]) == 1
    assert pic_deps.main(["install", "--script", str(script)]) == 0
    assert root.is_dir()
    assert pic_deps.main(["check", "--script", str(script)]) == 0
