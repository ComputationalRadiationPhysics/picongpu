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
    # The workflow input must provide a stable destination (the run dir) and
    # the generated submit script must use it (not the step's working
    # directory, which is the cwltool job cache dir); $(pwd -P) may only
    # remain as the fallback for standalone runs.
    runner = sim.picongpu_get_runner()
    with runner.workflow_input_path.open("r") as file:
        workflow_input = json.load(file)
    assert workflow_input["destination_path"] == str(runner.run_dir)

    submission_script = runner.submission_script_path.read_text()
    # the results link is written against the stable destination, and the
    # destination itself defaults to the passed value with $(pwd -P) only
    # as the standalone fallback
    assert "ln -s $destination_path/simOutput" in submission_script
    assert 'destination_path="${2:-$(pwd -P)}"' in submission_script


# ---------------------------------------------------------------------------
# End-to-end regression (m2): run the *actual* generated submit.sh together
# with the real organize_output step inside a dummy 2-step cwltool workflow
# (same workdir layout as submit.cwl, same RuntimeContext as runner.run()).
#
# The fake batch file has the structure of the real templates: the job cd's
# to $TBG_dstPath and executes $TBG_dstPath/input/bin/picongpu. tbg is
# simulated by resolving !TBG_dstPath to a per-step cache path at prepare
# time. This is the guard that would have caught:
#   - C1: the in-workflow job running from a directory whose input/ has not
#     been staged yet (organize_output runs *after* submit).
#   - M1: leftover resolved !TBG_dstPath references that are not rewritten.
# It needs neither a GPU nor a PIConGPU build (a fake executable is used).
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
  destination_path:
    type: string?
    inputBinding:
      position: 3
    default: null
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


def _wait_for_any(paths, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        for path in paths:
            if path.exists():
                return True
        time.sleep(0.1)
    return False


def test_in_workflow_job_runs_from_stable_destination(tmp_path):
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

    # simulate tbg: the resolved submit.start has !TBG_dstPath baked in at
    # prepare time as a per-step cwltool job cache path (shape of the real one)
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
  destination_path:
    type: string?
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
      destination_path: destination_path
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
        "destination_path": str(run_dir),
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

    # the default (bash) submit system backgrounds the job; wait for it to
    # produce a result. Waiting on the pid is unreliable here (the orphaned
    # background job can linger as an unreaped zombie, so os.kill(pid, 0)
    # keeps succeeding), so wait on its output instead: simOutput/output on
    # success or job_failed.txt if the executable was not found.
    sim_output = run_dir / "simOutput" / "output"
    job_failed = run_dir / "job_failed.txt"
    assert _wait_for_any([sim_output, job_failed]), "in-workflow job did not produce a result"

    # (b) the in-workflow job actually ran: it found and executed the binary
    # from the stable destination and wrote simOutput there (not a cache dir)
    assert not job_failed.exists()
    assert sim_output.is_file()
    assert "full simulation time:" in sim_output.read_text()

    # (a) no .cwl_cache references remain in the final outputs
    for name in ("tbg/submit.start", "link_results.sh", "submission_information.txt"):
        content = (run_dir / name).read_text()
        assert ".cwl_cache" not in content, f"{name} still references .cwl_cache:\n{content}"
    leaked = [
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file() and ".cwl_cache" not in p.parts and b".cwl_cache" in p.read_bytes()
    ]
    assert not leaked, f".cwl_cache leaked into final outputs: {leaked}"

    # the stable references point at where the results actually are
    assert f'TBG_dstPath="{run_dir}"' in (run_dir / "tbg" / "submit.start").read_text()
    assert f"ln -s {run_dir}/simOutput" in (run_dir / "link_results.sh").read_text()
