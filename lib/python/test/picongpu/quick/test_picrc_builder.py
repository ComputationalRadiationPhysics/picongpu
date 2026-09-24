"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import picongpu.picrc_builder as picrc_builder
from picongpu.picrc_builder import _ask, _editable_keys, _require_selection, resolve_target_path, write_output


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


def test_require_selection_accepts_non_empty():
    assert _require_selection(["any_key"]) is True


def test_require_selection_rejects_empty():
    # questionary.checkbox confirms on <enter> even with nothing toggled, so an
    # empty selection must be rejected rather than silently skipping the edits.
    assert _require_selection([]) is not True


def test_ask_uses_unsafe_ask():
    # `_ask` must not use questionary's `ask()`, which swallows KeyboardInterrupt.
    calls = []

    class DummyQuestion:
        def unsafe_ask(self):
            calls.append("unsafe_ask")
            return "answer"

        def ask(self):  # pragma: no cover - must never be used
            calls.append("ask")
            raise AssertionError("_ask must not call Question.ask()")

    assert _ask(DummyQuestion()) == "answer"
    assert calls == ["unsafe_ask"]


def test_main_aborts_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    def raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(picrc_builder, "_run", raise_interrupt)
    assert picrc_builder.main([]) is None
    assert "Aborted" in capsys.readouterr().out


def test_editable_keys_excludes_only_internal_bookkeeping():
    keys = ["pic_backend", "pic_src_path", "required_information", "tbg_partition"]
    assert _editable_keys(keys) == ["pic_backend", "pic_src_path", "tbg_partition"]


def test_editable_keys_keeps_required_and_optional_parameters():
    # Re-editing required parameters is allowed; only `required_information` is internal.
    keys = ["pic_backend", "tbg_partition", "scratch_dir"]
    assert _editable_keys(keys) == keys
