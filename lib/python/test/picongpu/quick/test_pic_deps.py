"""
Tests for pic-deps (lib/python/picongpu/pic_deps.py).
"""

from pathlib import Path

import pytest

from picongpu._rc_params import PRESET_STORAGE_PATH, RCParams
from picongpu.pic_deps import parse_guard_roots, resolve_preset_script, check_dependencies


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


def test_resolve_preset_script_missing_script():
    # bash preset has no dependencies_autoinstall.sh -> unsupported
    p = RCParams(preset="bash/bash_picongpu.profile.example")
    with pytest.raises(SystemExit, match="dependencies_autoinstall.sh"):
        resolve_preset_script(p)


def test_check_dependencies_reports_all_missing_for_empty_piclibs():
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    results = check_dependencies(_completed_params())
    roots = [r for r, _, _ in results]
    assert roots == ["BOOST_ROOT", "BLOSC_ROOT", "PNGwriter_ROOT", "ADIOS2_ROOT", "OPENPMD_ROOT"]
    # pic_libs points at a non-existent dir -> every dep is missing
    assert all(ok is False for _, _, ok in results)
    # resolved paths are the profile-derived versioned roots (non-empty)
    assert all(resolved for _, resolved, _ in results)
