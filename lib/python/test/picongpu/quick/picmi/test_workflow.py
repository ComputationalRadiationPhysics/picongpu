"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
import subprocess

from cwltool.context import RuntimeContext
from cwltool.factory import Factory, WorkflowStatus
from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation
from picongpu.templates import path as tpath
from pytest import fixture, raises


@fixture
def sim():
    number_of_cells = 32
    cell_size = 1
    sim = Simulation(
        time_step_size=17,
        max_steps=4,
        solver=ElectromagneticSolver(
            method="Yee",
            grid=Cartesian3DGrid(
                number_of_cells=[number_of_cells, number_of_cells, number_of_cells],
                lower_bound=[0, 0, 0],
                upper_bound=list(map(lambda x: number_of_cells * x, [cell_size, cell_size, cell_size])),
                # required, otherwise won't spawn
                lower_boundary_conditions=["open", "open", "periodic"],
                upper_boundary_conditions=["open", "open", "periodic"],
            ),
        ),
    )
    sim.picongpu_get_runner().generate()
    return sim


@fixture
def workflow_definition_path(sim):
    return sim.picongpu_get_runner().workflow_definition_path


@fixture
def workflow_input(sim):
    with sim.picongpu_get_runner().workflow_input_path.open("r") as file:
        return json.load(file)


def test_validate_workflow(workflow_definition_path, workflow_input):
    # Couldn't have come up with a stranger interface:
    # The `validate_only` mode of the factory uses an exception to shortcircuit apparently.
    # Well, in this case "success" means:
    with raises(WorkflowStatus, match="Completed ValidationSuccess"):
        Factory(runtime_context=RuntimeContext(kwargs={"validate_only": True})).make(str(workflow_definition_path))(
            **workflow_input
        )


def run_organize_output(tmp_path, submit_start_content, link_results_content):
    """Run the real organize_output.sh against fake submit-step outputs."""
    organize_script = tpath() / "workflow" / "scripts" / "organize_output.sh"

    project_path = tmp_path / "project"
    project_path.mkdir()
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()

    tbg_directory = tmp_path / "tbg"
    tbg_directory.mkdir()
    (tbg_directory / "submit.start").write_text(submit_start_content)

    submission_information = tmp_path / "submission_information.txt"
    submission_information.write_text("12345\n")

    link_results_script = tmp_path / "link_results.sh"
    link_results_script.write_text(link_results_content)

    workdir = tmp_path / "work"
    workdir.mkdir()
    subprocess.run(
        [
            "bash",
            str(organize_script),
            str(project_path),
            str(bin_directory),
            str(tbg_directory),
            str(submission_information),
            str(link_results_script),
        ],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    return workdir


def test_organize_output_rewrites_cwl_cache_references(tmp_path):
    # The submit step runs inside cwltool's per-step job cache directory
    # (<run_dir>/.cwl_cache/<md5>) and bakes that path into tbg/submit.start
    # (TBG_dstPath, --chdir) and link_results.sh.
    # organize_output must rewrite those references to the stable run directory.
    run_dir = tmp_path / "run"
    cache_dir = run_dir / ".cwl_cache" / "79c1166a1abedf768c358bf7d959ef0d"

    workdir = run_organize_output(
        tmp_path,
        submit_start_content=f"TBG_dstPath={cache_dir}\n#SBATCH --chdir={cache_dir}\necho job\n",
        link_results_content=f"ln -s {cache_dir}/simOutput $1\n",
    )

    submit_start = (workdir / "tbg" / "submit.start").read_text()
    assert f"TBG_dstPath={run_dir}\n" in submit_start
    assert f"--chdir={run_dir}\n" in submit_start
    assert ".cwl_cache" not in submit_start

    link_results = (workdir / "link_results.sh").read_text()
    assert f"ln -s {run_dir}/simOutput $1\n" in link_results
    assert ".cwl_cache" not in link_results


def test_organize_output_is_noop_without_cwl_cache_references(tmp_path):
    # Files without cache references must pass through unchanged,
    # and the script must not fail (set -euxo pipefail).
    workdir = run_organize_output(
        tmp_path,
        submit_start_content="TBG_dstPath=/some/stable/dir\necho job\n",
        link_results_content="ln -s /some/stable/dir/simOutput $1\n",
    )

    assert (workdir / "tbg" / "submit.start").read_text() == "TBG_dstPath=/some/stable/dir\necho job\n"
    assert (workdir / "link_results.sh").read_text() == "ln -s /some/stable/dir/simOutput $1\n"


def test_submission_uses_stable_destination_path(sim):
    # The workflow input must provide a stable destination (the run dir)
    # and the generated submit script must use it instead of the step's
    # working directory (the cwltool job cache dir).
    runner = sim.picongpu_get_runner()
    with runner.workflow_input_path.open("r") as file:
        workflow_input = json.load(file)
    assert workflow_input["destination_path"] == str(runner.run_dir)

    submission_script = runner.submission_script_path.read_text()
    assert "TBG_dstPath=$destination_path" in submission_script
    assert "ln -s $destination_path/simOutput" in submission_script
    # $(pwd -P) may only remain as the fallback for standalone runs
    assert submission_script.count("$(pwd -P)") == 1
