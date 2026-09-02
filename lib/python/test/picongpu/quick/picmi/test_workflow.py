"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
import subprocess
import time

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


def run_organize_output(tmp_path, submit_start_content, link_results_content, project_files=None):
    """Run the real organize_output.sh against fake submit-step outputs.

    project_files: optional {relative_path: content} files to place in the
    project directory; the step copies the project into input/.
    """
    organize_script = tpath() / "workflow" / "scripts" / "organize_output.sh"

    project_path = tmp_path / "project"
    project_path.mkdir()
    for rel, content in (project_files or {}).items():
        file = project_path / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
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


def test_organize_output_strips_cwl_cache_references_except_link_results(tmp_path):
    # The submit step runs in isolation inside cwltool's per-step job cache
    # directory (<run_dir>/.cwl_cache/<md5>) and bakes that path into its
    # outputs (tbg/submit.start: TBG_dstPath, --chdir; any other file that
    # embeds it). organize_output must strip the cache reference from every
    # generated file so the run_dir looks as if the simulation ran there
    # directly -- EXCEPT link_results.sh, which is the one file allowed to
    # keep pointing at the cache (that is where the isolated job wrote
    # its results).
    run_dir = tmp_path / "run"
    cache_dir = run_dir / ".cwl_cache" / "79c1166a1abedf768c358bf7d959ef0d"

    workdir = run_organize_output(
        tmp_path,
        submit_start_content=f"TBG_dstPath={cache_dir}\n#SBATCH --chdir={cache_dir}\necho job\n",
        link_results_content=f"ln -s {cache_dir}/simOutput $1\n",
        project_files={"N.cfg": f"# results live in {cache_dir}\n"},
    )

    # tbg/submit.start: cache ref stripped -> points at the run dir
    submit_start = (workdir / "tbg" / "submit.start").read_text()
    assert f"TBG_dstPath={run_dir}\n" in submit_start
    assert f"--chdir={run_dir}\n" in submit_start
    assert ".cwl_cache" not in submit_start

    # every other generated file is stripped too (here: input/N.cfg)
    cfg = (workdir / "input" / "N.cfg").read_text()
    assert f"# results live in {run_dir}\n" in cfg
    assert ".cwl_cache" not in cfg

    # link_results.sh is the ONLY exception: it keeps the cache reference
    link_results = (workdir / "link_results.sh").read_text()
    assert link_results == f"ln -s {cache_dir}/simOutput $1\n"
    assert ".cwl_cache" in link_results


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


def test_submit_step_runs_in_isolation(sim):
    # The agreed isolation model: the submit step runs in isolation in its own
    # (cwltool job-cache) working directory. It must NOT stage inputs into, or
    # rewrite paths towards, the final run directory (that would let a step
    # mutate outside data and break CWL step isolation); the run_dir is made
    # self-contained later, by the organize_output step. So there is no stable
    # destination plumbing, and the submit script resolves TBG_dstPath/--chdir
    # to its own working directory ($(pwd -P)).
    runner = sim.picongpu_get_runner()
    with runner.workflow_input_path.open("r") as file:
        workflow_input = json.load(file)
    assert "destination_path" not in workflow_input

    submission_script = runner.submission_script_path.read_text()
    assert "TBG_dstPath=$(pwd -P)" in submission_script
    assert "--chdir=$(pwd -P)" in submission_script
    # no stable-destination plumbing / input staging into the run directory
    assert "destination_path" not in submission_script
    assert "mkdir -p" not in submission_script


# ---------------------------------------------------------------------------
# End-to-end regression: run the *actual* generated submit.sh together with
# the real organize_output step inside a dummy 2-step cwltool workflow (same
# workdir layout as submit.cwl, same RuntimeContext as runner.run()).
#
# The agreed isolation model (PR #9 rework, see STYLE-GUIDE rule 17):
#   - the submit step runs in isolation in its own (cwltool job-cache) working
#     directory: it resolves TBG_dstPath/--chdir to $(pwd -P) and the
#     in-workflow job runs from that cache directory. It does NOT stage inputs
#     into, or reach into, the final run_dir.
#   - organize_output then strips every reference to that internal job cache
#     from all generated files *except* link_results.sh, which keeps pointing
#     at the cache (where the isolated job actually wrote its results).
# Afterwards the run_dir looks as if the simulation had run there directly,
# while CWL step isolation is preserved.
#
# The fake batch file has the structure of the real templates: the job cd's to
# $TBG_dstPath and executes $TBG_dstPath/input/bin/picongpu. tbg is simulated
# by resolving !TBG_dstPath to a per-step cache path at prepare time. Needs
# neither a GPU nor a PIConGPU build (a fake executable is used).
# ---------------------------------------------------------------------------

_FAKE_PICONGPU = """#!/bin/bash
mkdir -p simOutput
echo "full simulation time: 42" > simOutput/output
"""

# Structure of the real resolved batch files (bash/mpiexec.tpl etc.):
# the job cd's to $TBG_dstPath and runs the binary from $TBG_dstPath/input/bin.
_FAKE_SUBMIT_START = """#!/bin/bash
TBG_dstPath="__DST__"
cd $TBG_dstPath
if [ ! -x "$TBG_dstPath/input/bin/picongpu" ]; then
    echo "JOB-FAILED: executable missing at $TBG_dstPath/input/bin/picongpu" > job_failed.txt
    exit 1
fi
"$TBG_dstPath/input/bin/picongpu" || echo "JOB-FAILED: picongpu exit $?" > job_failed.txt
"""

_DUMMY_SUBMIT_CWL = """cwlVersion: v1.2
class: CommandLineTool
label: "dummy submit step (mirrors real submit.cwl workdir layout)"
requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: submit.sh
        entry: $(inputs.script)
      - entryname: input/bin
        entry: $(inputs.bin_directory)
      - entryname: input/etc
        entry: $(inputs.etc_directory)
      - entryname: tbg_link
        entry: $(inputs.tbg_link)
  EnvVarRequirement:
    envDef:
      - envName: PICONGPU_RUNNING_AS_CWL
        envValue: "1"
baseCommand: ./submit.sh
inputs:
  script:
    type: File
  bin_directory:
    type: Directory
  etc_directory:
    type: Directory
  tbg_link:
    type: Directory
  submit_system:
    type: string?
    inputBinding:
      position: 2
    default: "bash"
outputs:
  submission_information:
    type: File
    outputBinding:
      glob: "submission_information.txt"
  link_results_script:
    type: File
    outputBinding:
      glob: "link_results.sh"
  tbg_directory:
    type: Directory
    outputBinding:
      glob: "tbg"
"""


def _wait_for_pred(pred, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_in_workflow_steps_isolated_and_run_dir_self_contained(tmp_path):
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
    runner = sim.picongpu_get_runner(setup_dir=tmp_path / "setup", run_dir=tmp_path / "run")
    runner.generate()
    run_dir = runner.run_dir

    # simulate tbg: the resolved submit.start has the prepare-time destination
    # baked in as a per-step cwltool job cache path (shape of the real one).
    # The isolated submit step rewrites TBG_dstPath to its own working dir.
    old_dst = str(tmp_path / ".cwl_cache" / "0123456789abcdef0123456789abcdef" / "run_dir")
    tbg_link = tmp_path / "tbg_link"
    tbg_link.mkdir()
    (tbg_link / "submit.start").write_text(_FAKE_SUBMIT_START.replace("__DST__", old_dst))
    (tbg_link / "submit.start").chmod(0o755)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    (fake_bin / "picongpu").write_text(_FAKE_PICONGPU)
    (fake_bin / "picongpu").chmod(0o755)
    fake_etc = tmp_path / "fake_etc"
    fake_etc.mkdir()
    (fake_etc / "N.cfg").write_text("fake cfg\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / "N.cfg").write_text("fake project cfg\n")

    submit_cwl = tmp_path / "dummy_submit.cwl"
    submit_cwl.write_text(_DUMMY_SUBMIT_CWL)
    organize_cwl = tpath() / "workflow" / "steps" / "organize_output.cwl"
    workflow_cwl = tmp_path / "dummy_workflow.cwl"
    workflow_cwl.write_text(
        f"""cwlVersion: v1.2
class: Workflow
inputs:
  script:
    type: File
  bin_directory:
    type: Directory
  etc_directory:
    type: Directory
  tbg_link:
    type: Directory
  project_path:
    type: Directory
  organize_script:
    type: File
outputs:
  input_directory:
    type: Directory
    outputSource: organize_output_step/input_directory
  tbg_directory:
    type: Directory
    outputSource: organize_output_step/tbg_directory
  link_results_script:
    type: File
    outputSource: organize_output_step/link_results_script
  submission_information:
    type: File
    outputSource: organize_output_step/submission_information
steps:
  submit_step:
    run: {submit_cwl}
    in:
      script: script
      bin_directory: bin_directory
      etc_directory: etc_directory
      tbg_link: tbg_link
    out: [submission_information, link_results_script, tbg_directory]
  organize_output_step:
    run: {organize_cwl}
    in:
      script: organize_script
      project_path: project_path
      bin_directory: bin_directory
      tbg_directory: submit_step/tbg_directory
      submission_information: submit_step/submission_information
      link_results_script: submit_step/link_results_script
    out: [input_directory, tbg_directory, link_results_script, submission_information]
"""
    )
    inputs = {
        "script": {"class": "File", "location": str(runner.submission_script_path)},
        "bin_directory": {"class": "Directory", "location": str(fake_bin)},
        "etc_directory": {"class": "Directory", "location": str(fake_etc)},
        "tbg_link": {"class": "Directory", "location": str(tbg_link)},
        "project_path": {"class": "Directory", "location": str(project)},
        "organize_script": {
            "class": "File",
            "location": str(tpath() / "workflow" / "scripts" / "organize_output.sh"),
        },
    }

    Factory(
        runtime_context=RuntimeContext(
            kwargs={
                "outdir": str(run_dir),
                "rm_tmpdir": False,
                "move_outputs": "copy",
                "cachedir": str(runner.cwl_cachedir),
                "preserve_entire_environment": True,
            }
        )
    ).make(str(workflow_cwl))(**inputs)

    # the default (bash) submit system backgrounds the isolated job, which
    # runs from its own cwltool job-cache working directory and writes
    # simOutput there (inside run_dir/.cwl_cache/...), not directly in the
    # run_dir. Waiting on the pid is unreliable (the orphaned background job
    # can linger as an unreaped zombie), so wait on its output instead:
    # simOutput/output on success or job_failed.txt if the executable was not
    # found.
    def _find(pattern):
        return next((p for p in run_dir.rglob(pattern) if p.is_file()), None)

    assert _wait_for_pred(lambda: _find("simOutput/output") or _find("job_failed.txt")), (
        "in-workflow job did not produce a result"
    )

    # (b) the isolated job actually ran: it found and executed the binary from
    # its own (job-cache) working directory and wrote simOutput there
    job_failed = _find("job_failed.txt")
    sim_output = _find("simOutput/output")
    assert job_failed is None, f"job failed:\n{job_failed.read_text() if job_failed is not None else ''}"
    assert sim_output is not None
    assert ".cwl_cache" in str(sim_output), "the isolated job wrote simOutput into its own job-cache dir"
    assert "full simulation time:" in sim_output.read_text()

    # (a) the final run_dir is self-contained: no generated file references
    # the internal job cache -- except link_results.sh, which keeps the
    # reference to where the isolated job actually wrote its results.
    submit_start = (run_dir / "tbg" / "submit.start").read_text()
    assert f"TBG_dstPath={run_dir}" in submit_start
    assert ".cwl_cache" not in submit_start
    assert ".cwl_cache" not in (run_dir / "submission_information.txt").read_text()
    link_results = (run_dir / "link_results.sh").read_text()
    assert ".cwl_cache" in link_results  # the one allowed exception
    leaked = [
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file()
        and p.name != "link_results.sh"
        and ".cwl_cache" not in p.parts
        and b".cwl_cache" in p.read_bytes()
    ]
    assert not leaked, f".cwl_cache leaked into final outputs: {leaked}"
