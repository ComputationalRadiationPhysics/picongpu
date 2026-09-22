"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from picongpu.picrc_builder import resolve_target_path, write_output


def test_resolve_none_is_new_target():
    assert resolve_target_path(None) == (None, True)


def test_resolve_non_existent_file_is_new_target():
    with TemporaryDirectory() as d:
        missing = Path(d) / "does_not_exist" / "picongpurc.toml"
        assert resolve_target_path(missing) == (missing, True)


def test_resolve_directory_points_into_directory():
    with TemporaryDirectory() as d:
        target, is_new = resolve_target_path(Path(d))
        assert target == Path(d) / "picongpurc.toml"
        assert is_new is True


def test_resolve_existing_file_is_loaded():
    with TemporaryDirectory() as d:
        existing = Path(d) / "existing.toml"
        existing.write_text('preset = "bash"\n')
        target, is_new = resolve_target_path(existing)
        assert target == existing
        assert is_new is False


def test_write_output_creates_missing_parent_dirs():
    with TemporaryDirectory() as d:
        deep = Path(d) / "a" / "b" / "c" / "picongpurc.toml"
        assert not deep.parent.exists()
        write_output({"preset": "bash", "author": "someone@example.com"}, deep)
        assert deep.exists()
        content = deep.read_text()
        assert "preset" in content
