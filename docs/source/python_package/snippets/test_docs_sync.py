"""
Tests that keep prose lists in the documentation in sync with the code.

This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

The ``configuring_environment`` and ``running_simulation`` pages describe
implementation details in prose (the ``profile_content`` precedence cascade
and the layout of the run directory). These are easy to get out of sync with
the code, so this module derives the facts from the implementation and checks
the documented statements against them.
"""

import re
import subprocess
from pathlib import Path

from picongpu import core
from picongpu._rc_params import PROFILE_PARAMETERS, RCParams

DOCS_SOURCE = Path(__file__).resolve().parents[2]
CONFIGURING_ENV = DOCS_SOURCE / "python_package/foundations/configuring_environment.rst"
RUNNING_SIMULATION = DOCS_SOURCE / "python_package/foundations/running_simulation.rst"
REPO_ROOT = DOCS_SOURCE.parents[1]


def test_preset_defaults_are_generated_from_code():
    """The preset-default list is generated from PROFILE_PARAMETERS, not written out."""
    text = CONFIGURING_ENV.read_text()
    assert ".. picongpu-preset-defaults::" in text, (
        "the preset-default list should be rendered by the picongpu-preset-defaults directive"
    )

    # the directive is registered in conf.py and consumes PROFILE_PARAMETERS
    conf = (DOCS_SOURCE / "conf.py").read_text()
    assert "picongpu-preset-defaults" in conf
    assert "PROFILE_PARAMETERS" in conf

    # at least one non-required parameter exists to be documented
    assert any(not parameter.is_required for parameter in PROFILE_PARAMETERS)


def _documented_profile_sources():
    """Return the ranked ``profile_content`` sources named in the docs, in order."""
    text = CONFIGURING_ENV.read_text()
    section = text.split("The ``profile_content`` is determined by the following cascade", 1)[1]
    section = section.split("The following list gives a redundant configuration", 1)[0]
    return re.findall(r"^\s+\d+\.\s+(.*)$", section, re.MULTILINE)


def test_documented_profile_cascade_matches_implementation(tmp_path):
    """Each documented source wins over all sources documented as lower-precedence."""
    sources = _documented_profile_sources()
    assert len(sources) == 5, f"expected 5 documented profile sources, got {sources}"

    profile_path = tmp_path / "profile"
    profile_path.write_text("content of profile_path")
    template_path = tmp_path / "profile-template"
    template_path.write_text("content of profile_template_path")

    # one value per documented level, in the documented order (highest first)
    levels = [
        RCParams(profile_content="content of profile_content"),
        RCParams(profile_path=str(profile_path)),
        RCParams(profile_template_content="content of profile_template_content"),
        RCParams(profile_template_path=str(template_path)),
        RCParams(),
    ]
    assert "profile_content" in sources[0]
    assert "profile_path" in sources[1]
    assert "profile_template_content" in sources[2]
    assert "profile_template_path" in sources[3]
    assert "PATH" in sources[4]

    assert levels[0].profile_content == "content of profile_content"
    assert levels[1].profile_content == "content of profile_path"
    assert levels[2].profile_content == "content of profile_template_content"
    assert levels[3].profile_content == "content of profile_template_path"
    assert levels[4].profile_content == f'export PATH="{core.path("bin")}:$PATH"'

    # the documented ordering is a strict cascade: a higher level always wins
    # over every lower one, regardless of which lower entries are also set
    for winner_index, winner in enumerate(levels):
        for lower in levels[winner_index + 1 :]:
            merged = RCParams(**(winner.model_dump() | lower.model_dump()))
            assert merged.profile_content == winner.profile_content


def _documented_run_directory_entries():
    """Return the file/directory names of the documented run directory tree."""
    text = RUNNING_SIMULATION.read_text()
    lines = text.split("the run directory looks like this::", 1)[1].splitlines()
    entries = set()
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("my_run"):
            started = True
            continue
        if not started:
            continue
        if not stripped:
            break
        name = line.strip().lstrip("\u2502\u251c\u2514\u2500 ").split("#", 1)[0].strip().rstrip("/")
        if name:
            entries.add(name)
    return entries


def test_documented_run_directory_matches_workflow_outputs(tmp_path):
    """Every entry of the documented run directory is produced by the workflow.

    The ``organize_output.sh`` workflow step is executed for real (with dummy
    inputs) and the resulting directory tree is compared against the tree
    documented in ``running_simulation.rst``.
    """
    documented = _documented_run_directory_entries()
    assert documented, "failed to parse the documented run directory tree"

    # dummy inputs for the organize_output step
    project_path = tmp_path / "project"
    for entry in ("etc", "include", "metadata", "workflow"):
        (project_path / entry).mkdir(parents=True)
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    tbg_directory = tmp_path / "tbg_input"
    tbg_directory.mkdir()
    for name in ("submit.start", "submit.tpl", "submit.cfg"):
        (tbg_directory / name).write_text("")
    submission_information = tmp_path / "submission_information.txt"
    submission_information.write_text("")
    link_results = tmp_path / "link_results.sh"
    link_results.write_text("#!/bin/bash\n")

    work_dir = tmp_path / "run"
    work_dir.mkdir()
    script = REPO_ROOT / "lib/python/picongpu/templates/workflow/scripts/organize_output.sh"
    subprocess.run(
        [
            "bash",
            str(script),
            str(project_path),
            str(bin_directory),
            str(tbg_directory),
            str(submission_information),
            str(link_results),
        ],
        cwd=work_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    produced = set()
    for path in work_dir.rglob("*"):
        rel = path.relative_to(work_dir)
        produced.add(rel.name)
        produced.add(str(rel).split("/")[0])
    # the cache directory is created by the runner, not the workflow step
    produced.add(".cwl_cache")

    missing = documented - produced
    assert not missing, f"documented run-directory entries not produced by the workflow: {sorted(missing)}"
