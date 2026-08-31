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

Every TOML snippet (``.picongpurc.toml`` examples) is parsed with ``tomllib``
and then applied for real: a subprocess with an isolated ``HOME`` and
``PIC_RC`` pointed at the snippet file imports the PIConGPU python package,
and the resulting ``rc_params`` content is checked.

Every bash snippet is syntax-checked with ``bash -n`` in this suite.
The ``docs-snippets`` CI job (see ``.gitlab-ci.yml``) additionally executes
one bash flow for real: setup generation with the ``bash`` preset and
sourcing of the generated profile, as ``running_simulation/legacy_workflow.sh``
performs it (``share/ci/docs_snippets_profile_check.sh``).
No other bash snippet - in particular the ``cwltool``, ``pic-build`` and
``tbg`` invocations - is executed in CI.
"""

import json
import os
import re
import subprocess
import sys
import tomllib
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
TOML_SNIPPETS = sorted(SNIPPETS_DIR.glob("**/*.toml"))

TOML_EXPECTED = {
    "configuring_environment/rc_params_minimal.toml": {
        "preset": "bash",
    },
    "configuring_environment/rc_params_finetune_preset.toml": {
        "preset": "rosi-hzdr",
        # the toml value overrides the preset default (gpu-v100)
        "tbg_partition": "a100",
    },
    "configuring_environment/rc_params_shebang.toml": {
        "shebang": "#!/usr/bin/env zsh",
    },
    "configuring_environment/rc_params_profile_precedence.toml": {
        "my_rc_params_value": "Rendering template content directly",
        "profile_content": "echo 'Using profile_content directly'",
        "profile_path": "/path/to/my/profile",
        "profile_template_content": "echo {my_rc_params_value}",
        "profile_template_path": "/path/to/my/profile-template",
    },
}

EXPECTED_FILES = {
    "configuring_environment/rc_params_basic.py": {
        "stdout_contains": ["It worked!"],
    },
    "configuring_environment/rc_params_preset_guard.py": {
        "stderr_contains": ["triggered resetting rc_params"],
    },
    "configuring_environment/rc_params_list_presets.py": {
        # single-profile systems are presets by themselves,
        # multi-profile systems contribute one preset per profile example file
        "stdout_contains": [
            "bash",
            "rosi-hzdr",
            "jupiter-jsc",
            "hemera-hzdr/gpu_picongpu.profile.example",
            "zih-tud/A100_picongpu.profile.example",
        ],
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
    "quickstart/my_first_simulation.py": {
        "no_run": True,
        "files": [
            "my_first_simulation_setup/include/picongpu/param/simulation.param",
            "my_first_simulation_setup/workflow/workflow.cwl",
            "my_first_simulation_setup/workflow/scripts/picongpu.profile",
            "my_first_simulation_setup/metadata/pypicongpu_runner.json",
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
    "defining_simulation/warm_plasma.py": {
        "no_run": True,
        "files": [
            "warm_plasma_setup/include/picongpu/param/simulation.param",
            "warm_plasma_setup/include/picongpu/param/speciesDefinition.param",
            "warm_plasma_setup/include/picongpu/param/speciesInitialization.param",
            "warm_plasma_setup/workflow/workflow.cwl",
            "warm_plasma_setup/metadata/pypicongpu_runner.json",
        ],
        "file_contains": [
            ("warm_plasma_setup/include/picongpu/param/speciesDefinition.param", "ions"),
            ("warm_plasma_setup/include/picongpu/param/speciesDefinition.param", "electrons"),
        ],
    },
    "defining_simulation/laser_variants.py": {
        "no_run": True,
        "files": [
            "laser_variants_setup/include/picongpu/param/incidentField.param",
            "laser_variants_setup/workflow/workflow.cwl",
            "laser_variants_setup/metadata/pypicongpu_runner.json",
        ],
        "file_contains": [
            ("laser_variants_setup/include/picongpu/param/incidentField.param", "PyPIConGPUGaussianPulseParam"),
            ("laser_variants_setup/include/picongpu/param/incidentField.param", "PyPIConGPUDispersivePulseParam"),
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
    "selected_topics/time_steps.py": {
        "stdout_contains": [
            "slice(None, None, 10)",
            "slice(None, 5, None)",
            "slice(49, None, None)",
            "slice(1e-15, 5e-15, 2e-16)",
            "combined unit system: mixed",
            "It worked!",
        ],
    },
    "selected_topics/phase_space.py": {
        "no_run": True,
        "files": [
            "phase_space_setup/etc/picongpu/N.cfg",
            "phase_space_setup/workflow/workflow.cwl",
        ],
        "file_contains": [
            ("phase_space_setup/etc/picongpu/N.cfg", "--electrons_phaseSpace.period 0:-1:10"),
            ("phase_space_setup/etc/picongpu/N.cfg", "--electrons_phaseSpace.space y"),
            ("phase_space_setup/etc/picongpu/N.cfg", "--electrons_phaseSpace.momentum py"),
            # momentum range in units of m_species*c (see the phase_space page)
            ("phase_space_setup/etc/picongpu/N.cfg", "--electrons_phaseSpace.min -1.0"),
            ("phase_space_setup/etc/picongpu/N.cfg", "--electrons_phaseSpace.max 1.0"),
        ],
    },
    "selected_topics/energy_histogram.py": {
        "no_run": True,
        "files": [
            "energy_histogram_setup/etc/picongpu/N.cfg",
        ],
        "file_contains": [
            ("energy_histogram_setup/etc/picongpu/N.cfg", "--electrons_energyHistogram.period 0:-1:10"),
            ("energy_histogram_setup/etc/picongpu/N.cfg", "--electrons_energyHistogram.binCount 50"),
            ("energy_histogram_setup/etc/picongpu/N.cfg", "--electrons_energyHistogram.maxEnergy 500.0"),
        ],
    },
    "selected_topics/macro_particle_count.py": {
        "no_run": True,
        "files": [
            "macro_particle_count_setup/etc/picongpu/N.cfg",
        ],
        "file_contains": [
            ("macro_particle_count_setup/etc/picongpu/N.cfg", "--electrons_macroParticlesCount.period 0:-1:10"),
        ],
    },
    "selected_topics/openpmd.py": {
        "no_run": True,
        "files": [
            "openpmd_setup/etc/picongpu/N.cfg",
            "openpmd_setup/include/picongpu/param/fileOutput.param",
        ],
        "file_contains": [
            ("openpmd_setup/etc/picongpu/N.cfg", "--openPMD.pluginConfig"),
            ("openpmd_setup/include/picongpu/param/fileOutput.param", "FieldE"),
        ],
        "stdout_contains": [
            'file = "simData"',
            'file = "magneticField"',
            '"electrons"',
            '"E"',
            "kineticEnergy",
        ],
    },
    "selected_topics/binning.py": {
        "no_run": True,
        "files": [
            "binning_setup/include/picongpu/param/binningSetup.param",
        ],
        "file_contains": [
            ("binning_setup/include/picongpu/param/binningSetup.param", "gammaDistribution"),
            ("binning_setup/include/picongpu/param/binningSetup.param", "addParticleBinner"),
            ("binning_setup/include/picongpu/param/binningSetup.param", 'setNotifyPeriod("0:-1:10")'),
            # the filtered-species binner renders the filter as a boolean functor
            ("binning_setup/include/picongpu/param/binningSetup.param", "fastGammaDistribution"),
            ("binning_setup/include/picongpu/param/binningSetup.param", "FilteredSpecies"),
            ("binning_setup/include/picongpu/param/binningSetup.param", "Ekin > 1.6e-15"),
        ],
    },
    "selected_topics/radiation.py": {
        "no_run": True,
        "files": [
            "radiation_setup/etc/picongpu/N.cfg",
            "radiation_setup/include/picongpu/param/radiation.param",
        ],
        "file_contains": [
            ("radiation_setup/etc/picongpu/N.cfg", "--electrons_radiation.period 2:-1:5"),
            ("radiation_setup/etc/picongpu/N.cfg", "--electrons_radiation.totalRadiation"),
            ("radiation_setup/etc/picongpu/N.cfg", "--electrons_radiation.dump 5"),
        ],
    },
    "selected_topics/checkpoint.py": {
        "no_run": True,
        "files": [
            "checkpoint_setup/etc/picongpu/N.cfg",
        ],
        "file_contains": [
            ("checkpoint_setup/etc/picongpu/N.cfg", "--checkpoint.period 0:-1:20"),
            ("checkpoint_setup/etc/picongpu/N.cfg", "--checkpoint.directory checkpoints"),
        ],
    },
    "troubleshooting/validate_before_submit.py": {
        "files": [
            "validated_setup/workflow/workflow.cwl",
            "validated_setup/metadata/pypicongpu_runner.json",
        ],
        "stdout_contains": ["Input files generated in"],
    },
    "selected_topics/interactions.py": {
        "no_run": True,
        "files": [
            "adk_setup/include/picongpu/param/speciesDefinition.param",
            "bsi_setup/include/picongpu/param/speciesDefinition.param",
            "synchrotron_setup/include/picongpu/param/synchrotron.param",
            "synchrotron_setup/workflow/workflow.cwl",
        ],
        "file_contains": [
            ("adk_setup/include/picongpu/param/speciesDefinition.param", "ADKLinPol"),
            ("bsi_setup/include/picongpu/param/speciesDefinition.param", "BSIStarkShifted"),
            ("synchrotron_setup/include/picongpu/param/speciesDefinition.param", "synchrotron<species_photons>"),
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


@pytest.mark.parametrize("snippet", TOML_SNIPPETS, ids=lambda path: str(path.relative_to(SNIPPETS_DIR)))
def test_toml_snippet(snippet, tmp_path):
    expected = TOML_EXPECTED[snippet.relative_to(SNIPPETS_DIR).as_posix()]

    # the rendered file must be valid TOML carrying the documented values
    data = tomllib.loads(snippet.read_text())
    for key, value in expected.items():
        assert data[key] == value, f"{snippet.name}: expected {key!r} == {value!r}, got {data.get(key)!r}"

    # the exact rendered file must be picked up via PIC_RC
    # and produce the expected rc_params content
    home = tmp_path / "home"
    home.mkdir()

    environment = {
        key: value for key, value in os.environ.items() if key not in ("PIC_RC", "XDG_CONFIG_HOME", "XDG_DATA_HOME")
    }
    environment["HOME"] = str(home)
    environment["PIC_RC"] = str(snippet)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json\n"
            "from picongpu import rc_params\n"
            "print(json.dumps({key: str(value) for key, value in rc_params.items()}))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"snippet {snippet} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    applied = json.loads(result.stdout)
    for key, value in expected.items():
        assert applied[key] == str(value), (
            f"expected rc_params[{key!r}] == {value!r} after loading {snippet.name}, got {applied.get(key)!r}"
        )
