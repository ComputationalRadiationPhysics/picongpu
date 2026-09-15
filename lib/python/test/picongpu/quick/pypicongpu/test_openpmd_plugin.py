"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from picongpu.pypicongpu.output.openpmd_plugin import FieldDump, OpenPMDPlugin
from picongpu.pypicongpu.output.timestepspec import Spec, TimeStepSpec


def _plugin():
    return OpenPMDPlugin(
        sources=[(TimeStepSpec([Spec(start=0, stop=10, step=1)]), FieldDump(name="E", filtername=None, functor=None))]
    )


def test_config_filename_is_content_hash_only():
    plugin = _plugin()

    filename = plugin.config_filename(plugin._config_content)

    assert re.fullmatch(r"openPMD_config_[0-9a-f]{64}\.toml", filename)


def test_runtime_config_filename_is_cwd_relative():
    # The batch job runs with its CWD at <run_dir>/simOutput, while the config is
    # written to <run_dir>/input/etc. The rendered pluginConfig string is the path
    # of the config relative to that CWD, so it is stable across run dirs and
    # presets and carries no absolute location.
    plugin = _plugin()

    serialized = plugin._get_serialized()

    assert serialized["type_openPMD"] is True
    assert serialized["config_filename"].startswith("../input/etc/openPMD_config_")
    assert serialized["config_filename"].endswith(".toml")
    # the rendered string references the same content-hash filename
    assert Path(serialized["config_filename"]).name == plugin.config_filename(plugin._config_content)


def test_write_config_file_lands_in_render_root_etc():
    plugin = _plugin()
    filename = plugin.config_filename(plugin._config_content)

    with TemporaryDirectory() as tmpdir:
        written = plugin.write_config_file(Path(tmpdir))

        assert written == Path(tmpdir) / "etc" / filename
        assert written.is_file()
        assert written.read_text().splitlines()[0] == 'file = "simData"'


def test_get_serialized_is_pure():
    # serializing must be a pure function of the model: it performs no file I/O
    # (the old hack wrote the TOML and leaked a TemporaryDirectory here) and two
    # equal plugins serialize to an identical result.
    a = _plugin()
    b = _plugin()

    assert a == b
    assert a._get_serialized() == b._get_serialized()
    assert a._get_serialized() == a._get_serialized()


def test_no_setup_dir_state_or_leaked_tempdir():
    # the openPMD plugin no longer carries a (lazily auto-created, leaked)
    # setup_dir; that layout concern now lives in the Runner.
    plugin = _plugin()

    assert not hasattr(plugin, "setup_dir")
    assert not hasattr(plugin, "_setup_dir")
    assert not hasattr(plugin, "_setup_dir_explicit")
