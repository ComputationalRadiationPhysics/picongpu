"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from cwltool.context import RuntimeContext
from cwltool.factory import Factory, WorkflowStatus
from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation
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


@fixture
def generated_runner():
    """A Runner whose setup has been fully generated into a controlled run dir."""
    sim = Simulation(
        time_step_size=17,
        max_steps=4,
        solver=ElectromagneticSolver(
            method="Yee",
            grid=Cartesian3DGrid(
                number_of_cells=[32, 32, 32],
                lower_bound=[0, 0, 0],
                upper_bound=[32, 32, 32],
                lower_boundary_conditions=["open", "open", "periodic"],
                upper_boundary_conditions=["open", "open", "periodic"],
            ),
        ),
    )
    with TemporaryDirectory() as tmp:
        runner = sim.picongpu_get_runner(run_dir=Path(tmp) / "run")
        runner.generate()
        yield runner


@fixture
def fake_bin_directory():
    """A stand-in for the compiled binaries pic-build would produce (no C++ needed)."""
    with TemporaryDirectory() as tmp:
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir()
        (bin_dir / "picongpu.1").write_text("fake-binary")
        yield bin_dir


@fixture
def fake_tbg_directory():
    with TemporaryDirectory() as tmp:
        tbg_dir = Path(tmp) / "tbg"
        tbg_dir.mkdir()
        (tbg_dir / "submit.start").write_text("fake-tbg")
        yield tbg_dir


def test_organize_output_stages_input_into_run_dir(generated_runner, fake_bin_directory, fake_tbg_directory):
    """The generate->run CWL change stages the generated setup as ``input`` and
    emits ``input_directory`` exactly at ``run_dir/input`` (no shadowing of the
    just-generated setup). Exercises the ``entryname: input`` + ``cp -r bin
    input/bin`` path without needing a C++ build."""
    runner = generated_runner
    run_dir = runner.run_dir
    setup_dir = runner.setup_dir
    # sanity: generation produced the input tree (single self-contained run dir)
    assert (setup_dir / "include").is_dir()
    assert (setup_dir / "etc").is_dir()

    with NamedTemporaryFile("w", delete=False) as sub_info:
        sub_info.write("12345")
        sub_info_path = sub_info.name
    with NamedTemporaryFile("w", delete=False) as link:
        link.write("#!/bin/bash echo link")
        link_path = link.name
    try:
        result = Factory(
            runtime_context=RuntimeContext(
                kwargs={
                    "outdir": str(run_dir),
                    "rm_tmpdir": False,
                    "move_outputs": "copy",
                    "preserve_entire_environment": True,
                }
            )
        ).make(str(runner.workflow_dir_path / "steps" / "organize_output.cwl"))(
            script={"class": "File", "location": str(runner.workflow_scripts_path / "organize_output.sh")},
            project_path={"class": "Directory", "location": str(setup_dir)},
            bin_directory={"class": "Directory", "location": str(fake_bin_directory)},
            tbg_directory={"class": "Directory", "location": str(fake_tbg_directory)},
            submission_information={"class": "File", "location": sub_info_path},
            link_results_script={"class": "File", "location": link_path},
        )
    finally:
        Path(sub_info_path).unlink(missing_ok=True)
        Path(link_path).unlink(missing_ok=True)

    # the input_directory output lands exactly at run_dir/input
    out_location = result["input_directory"]["location"]
    if out_location.startswith("file://"):
        out_location = out_location[len("file://") :]
    assert Path(out_location) == setup_dir
    # the generated setup is intact and the compiled bin/ was merged into it
    assert (setup_dir / "include").is_dir()
    assert (setup_dir / "etc").is_dir()
    assert (setup_dir / "bin" / "picongpu.1").is_file()
