"""
Pytest suite for the documentation snippets in this directory.

This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: opencode
License: GPLv3+

Every Python snippet is executed in a subprocess (fresh working directory,
isolated ``HOME`` and ``PIC_RC``) and must exit with code 0.
Snippets that call ``simulation.run()`` are executed with the workflow run
step replaced by a no-op (see ``run_snippet.py``), so that no compilation or
job submission is required; where the snippet reads simulation results, the
harness emulates the corresponding output files.
Per-snippet expected artifacts are checked afterwards.

Every bash snippet is syntax-checked with ``bash -n`` in this suite.
The ``docs-snippets`` CI job (see ``.gitlab-ci.yml``) additionally executes
one bash flow for real: setup generation with the ``bash`` preset and
sourcing of the generated profile, as ``running_simulation/legacy_workflow.sh``
performs it (``share/ci/docs_snippets_profile_check.sh``).
No other bash snippet - in particular the ``cwltool``, ``pic-build`` and
``tbg`` invocations - is executed in CI.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from run_snippet import (
    PEAK_COUNT,
    PEAK_FOCAL,
    PEAK_SIGMA,
    SCAN_FOCALS,
    emulated_electron_count,
    write_synthetic_energy_histogram,
)

SNIPPETS_DIR = Path(__file__).parent
RUN_SNIPPET = SNIPPETS_DIR / "run_snippet.py"

PYTHON_SNIPPETS = sorted(
    path
    for path in SNIPPETS_DIR.glob("**/*.py")
    if path.name not in ("test_snippets.py", "run_snippet.py", "conftest.py")
)
BASH_SNIPPETS = sorted(SNIPPETS_DIR.glob("**/*.sh"))

EXPECTED_FILES = {
    "configuring_environment/rc_params_basic.py": {
        "stdout_contains": ["It worked!"],
    },
    "configuring_environment/rc_params_preset_guard.py": {
        "stderr_contains": ["triggered resetting rc_params"],
    },
    "configuring_environment/rc_params_list_presets.py": {
        "stdout_contains": ["bash", "rosi-hzdr", "jupiter-jsc"],
    },
    "configuring_environment/rc_params_finetune_preset.py": {
        # the preset default is printed by the items() loop;
        # the value is adjusted to "a100" afterwards
        "stdout_contains": ["preset: rosi-hzdr", "tbg_partition: gpu-v100"],
    },
    "defining_simulation/minimal_example.py": {
        "no_run": True,
        "files": [
            "minimal_example_setup/include/picongpu/param/simulation.param",
            "minimal_example_setup/workflow/workflow.cwl",
            "minimal_example_setup/workflow/scripts/picongpu.profile",
            "minimal_example_setup/metadata/pypicongpu_runner.json",
        ],
    },
    "defining_simulation/lwfa_example.py": {
        "no_run": True,
        "files": [
            "lwfa_example_setup/include/picongpu/param/simulation.param",
            "lwfa_example_setup/include/picongpu/param/incidentField.param",
            "lwfa_example_setup/include/picongpu/param/speciesDefinition.param",
            "lwfa_example_setup/include/picongpu/param/fileOutput.param",
            "lwfa_example_setup/workflow/workflow.cwl",
            "lwfa_example_setup/metadata/pypicongpu_runner.json",
        ],
        "file_contains": [
            ("lwfa_example_setup/include/picongpu/param/speciesDefinition.param", "hydrogen"),
            ("lwfa_example_setup/include/picongpu/param/speciesDefinition.param", "electrons"),
        ],
    },
    "defining_simulation/serialize_simulation.py": {
        "files": ["electrons.json"],
        "stdout_contains": ["serialized simulation into", "It worked!"],
    },
    "defining_simulation/multiple_simulations.py": {
        "no_run": True,
        "files": [f"scan/focal_{focal:.1e}/setup/include/picongpu/param/simulation.param" for focal in SCAN_FOCALS]
        + [f"scan/focal_{focal:.1e}/setup/workflow/input.yaml" for focal in SCAN_FOCALS],
    },
    "defining_simulation/postprocess_histogram.py": {
        "files": ["electron_count.png"],
    },
    # mechanics test: the optimizer runs against the synthetic,
    # harness-defined landscape of run_snippet.py (no compiled simulation)
    "defining_simulation/optimize_focal_position.py": {
        "no_run": True,
        "stdout_regex": [
            (r"optimal focal position: ([0-9]+\.[0-9]+e-05)", "focal_position"),
            (r"maximal electron count: ([0-9]+)", "maximal_count"),
        ],
    },
}


def _make_synthetic_scan(tmp_path):
    """Create run directories with synthetic EnergyHistogram output (as a real scan would leave)."""
    run_dirs = []
    for focal in SCAN_FOCALS:
        run_dir = tmp_path / "scan" / f"focal_{focal:.1e}"
        write_synthetic_energy_histogram(run_dir, emulated_electron_count(run_dir))
        run_dirs.append(run_dir)
    return run_dirs


@pytest.mark.parametrize("snippet", PYTHON_SNIPPETS, ids=lambda path: str(path.relative_to(SNIPPETS_DIR)))
def test_python_snippet(snippet, tmp_path):
    expectations = EXPECTED_FILES[snippet.relative_to(SNIPPETS_DIR).as_posix()]

    argv = []
    if snippet.name == "postprocess_histogram.py":
        argv = [str(run_dir) for run_dir in _make_synthetic_scan(tmp_path)]

    home = tmp_path / "home"
    home.mkdir()

    environment = {
        key: value for key, value in os.environ.items() if key not in ("PIC_RC", "XDG_CONFIG_HOME", "XDG_DATA_HOME")
    }
    environment["HOME"] = str(home)
    # point PIC_RC at a non-existent file so that no runtime configuration
    # (and in particular no preset) is picked up from the environment
    environment["PIC_RC"] = str(tmp_path / "nonexistent.picongpurc.toml")
    environment["MPLBACKEND"] = "Agg"

    command = [sys.executable, str(RUN_SNIPPET)]
    if expectations.get("no_run"):
        command.append("--no-run")
    command.append(str(snippet))
    command.extend(argv)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"snippet {snippet} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    for pattern in expectations.get("files", []):
        assert (tmp_path / pattern).exists(), f"expected artifact {pattern} not found in {tmp_path}"

    for pattern, content in expectations.get("file_contains", []):
        assert content in (tmp_path / pattern).read_text(), f"expected {content!r} in {pattern}"

    for needle in expectations.get("stdout_contains", []):
        assert needle in result.stdout, f"expected {needle!r} in stdout:\n{result.stdout}"

    for needle in expectations.get("stderr_contains", []):
        assert needle in result.stderr, f"expected {needle!r} in stderr:\n{result.stderr}"

    for pattern, name in expectations.get("stdout_regex", []):
        match = re.search(pattern, result.stdout)
        assert match, f"expected {pattern!r} in stdout:\n{result.stdout}"
        value = float(match.group(1))
        if name == "focal_position":
            assert abs(value - PEAK_FOCAL) <= PEAK_SIGMA / 2, (
                f"optimizer did not converge near {PEAK_FOCAL}, got {value}"
            )
        if name == "maximal_count":
            assert value >= PEAK_COUNT - 5, f"expected a maximal electron count near {PEAK_COUNT}, got {value}"


@pytest.mark.parametrize("snippet", BASH_SNIPPETS, ids=lambda path: str(path.relative_to(SNIPPETS_DIR)))
def test_bash_snippet_syntax(snippet):
    result = subprocess.run(["bash", "-n", str(snippet)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed for {snippet}:\n{result.stderr}"
