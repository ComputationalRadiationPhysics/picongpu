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


def test_check_dependencies_bash_enumerates_profile_roots():
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


def test_check_dependencies_conda_enumerates_shared_prefix_roots():
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


def test_check_dependencies_spack_enumerates_versioned_roots():
    if not Path("/bin/bash").is_file():
        pytest.skip("requires bash login shell")
    pic_libs = Path(__file__).parent / "nonexistent-piclibs"
    results = check_dependencies(_completed_bash_variant_params("bash/spack"))
    resolved = dict((root, path) for root, path, _ in results)
    assert resolved["BOOST_ROOT"] == str(pic_libs / "boost" / "1.87.0")
    assert resolved["HDF5_ROOT"] == str(pic_libs / "hdf5" / "2.0.0")
    assert resolved["PNGwriter_ROOT"] == str(pic_libs / "pngwriter" / "0.7.0")
