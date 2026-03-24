"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import datetime
import json
import logging
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from shutil import copy2, copytree
from typing import Annotated, Sequence

from cwltool.factory import Factory as WorkflowFactory
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    model_serializer,
)
from rocrate.rocrate import ROCrate

from picongpu import core, rc_params
from picongpu.templates import path as tpath

from .rendering import Renderer
from .simulation import Simulation
from .util import alt


def script_content_with(commands, working_dir: Path | None = None, rc_params=rc_params):
    if not isinstance(commands, str):
        commands = "\n".join(commands)
    cd = (
        f"""
# adjust working directory
cd {str(working_dir.absolute())}
"""
        if working_dir is not None
        else ""
    )
    return f"""
{rc_params.shebang}

{cd}

# preamble
{rc_params.preamble}

# profile content
{rc_params.profile_content}

# commands
{commands}
"""[1:]


def generate_bare_profile(path=None, rc_params=rc_params):
    if path is None:
        return generate_bare_profile(
            path=Path(tempfile.NamedTemporaryFile("w", delete=False, delete_on_close=False).name), rc_params=rc_params
        )
    if not isinstance(path, Path):
        return generate_bare_profile(path=Path(path), rc_params=rc_params)

    with rc_params.set_temporarily(preamble="", override_existing=False):
        with path.open("w") as file:
            file.write(script_content_with("", rc_params=rc_params))

    return path


def generate_bare_profile_as_in(script_path, path=None):
    if not isinstance(script_path, Path):
        return generate_bare_profile_as_in(Path(script_path).absolute(), path=path)
    if not script_path.is_absolute():
        return generate_bare_profile_as_in(script_path.absolute(), path=path)

    module_spec = spec_from_file_location("script", script_path)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return generate_bare_profile(path=path, rc_params=module.rc_params)


def get_tmpdir_with_name(name, parent: Path | None = None):
    """
    returns a not existing temporary directory path,
    which contains the given name
    :param name: part of the newly created directory name
    :param parent: if given: create the tmpdir there
    :return: not existing path to directory
    """
    with tempfile.TemporaryDirectory(
        prefix=f"pypicongpu-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{name}-", dir=parent
    ) as tmpdir:
        return Path(tmpdir).absolute()


class CmdLineFlags(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    @model_serializer(mode="wrap")
    def _compose_commandline(self, handler) -> str:
        return " ".join(map(_flag_to_str, handler(self).items()))


class CmdLine(BaseModel):
    @computed_field
    def command(self) -> str:
        raise NotImplementedError()

    args: Sequence[str]
    flags: CmdLineFlags

    @model_serializer(mode="plain")
    def compose_commandline(self) -> str:
        # example:
        #   model.command="cmd" model.args=('a', 'b'), model.flags={'x': 5, 'cfg': 6}
        #   cmd -x 5 --cfg 6 a b
        return f"{self.command} {self.flags.model_dump()} " + " ".join(self.args)


def _flag_to_str(flag_value_tuple) -> str:
    match flag_value_tuple:
        case (str(toggle), True):
            return f"-{toggle}"
        case (str(toggle), False):
            return ""
        case (_, None):
            return ""
        case (str(shorthand), value) if len(shorthand) == 1 and value is not None:
            return f"-{shorthand} {str(value)}"
        case (str(longoption), value) if len(longoption) > 1 and value is not None:
            return f"--{longoption} {str(value)}"
        case _:
            raise ValueError(f"Unknown serialization of {flag_value_tuple=} to string.")


class PicBuildFlags(CmdLineFlags):
    # We explicitly disallow the some shorthands like `-c`, `-t`, ...
    # because they overlap with tbg flags and could thus lead to confusion.
    jobs: int | None = Field(
        default=4,
        description="allow N jobs at once; infinite jobs if set to None",
        validation_alias=AliasChoices("jobs", "j"),
        serialization_alias="j",
    )

    cmake: str | None = Field(
        default=None,
        description=(
            'Extra arguments that are passed straight to CMake, e.g. "-DPIC_VERBOSE=21 -DCMAKE_BUILD_TYPE=Debug".'
        ),
        validation_alias=AliasChoices("cmake"),
        serialization_alias="c",
    )

    preset: int | None = Field(
        default=None,
        description="Configure this preset number from CMake flags.",
        ge=0,
        validation_alias=AliasChoices("preset"),
        serialization_alias="t",
    )

    force: bool = Field(
        default=False,
        description=("When set, clears the CMake file cache and forces a scan for new .param files."),
        validation_alias=AliasChoices("force", "f"),
        serialization_alias="f",
    )

    cmake_build_system: str | None = Field(
        default=None,
        description=("Select the build system used by CMake (e.g. ``Ninja``)."),
        validation_alias=AliasChoices("G"),
        serialization_alias="G",
    )

    help: bool = Field(
        default=False,
        description="Show the help message and exit.",
        validation_alias=AliasChoices("help", "h"),
        serialization_alias="h",
    )


class PicBuildCmdline(CmdLine):
    args: Sequence[str] = Field(default=tuple(), max_length=0)
    flags: PicBuildFlags = PicBuildFlags()

    @computed_field
    def command(self) -> str:
        return "pic-build"


class TBGFlags(CmdLineFlags):
    # We explicitly disallow the some shorthands like `-c`, `-t`, ...
    # because they overlap with pic-build flags and could thus lead to confusion.
    cfg_file: str = Field(
        default="etc/picongpu/N.cfg",
        description="Configuration file to set up batch file.",
        validation_alias=AliasChoices("cfg"),
        serialization_alias="c",
    )

    submit_system: str | None = Field(
        default="bash",
        description="Submit command (qsub, 'qsub -h', sbatch, ...).",
        validation_alias=AliasChoices("submit", "s"),
        serialization_alias="s",
    )

    template_file: str | None = Field(
        default="$TBG_TPLFILE",
        description="Template to create a batch file from.",
        validation_alias=AliasChoices("tpl"),
        serialization_alias="t",
    )

    overwrite_vars: list[str] | None = Field(
        default=None,
        description="Overwrite any template variable.",
        validation_alias=AliasChoices("o"),
        serialization_alias="o",
    )

    force: bool = Field(
        default=False,
        description="Override if 'destinationPath' exists.",
        validation_alias=AliasChoices("force", "f"),
        serialization_alias="f",
    )

    help: bool = Field(
        default=False,
        description="Show the help message and exit.",
        validation_alias=AliasChoices("help", "h"),
        serialization_alias="h",
    )


class TbgCmdLine(CmdLine):
    args: Sequence[str] = Field(min_length=1, max_length=2)
    flags: TBGFlags = TBGFlags()

    @computed_field
    def command(self) -> str:
        return "tbg"


class Runner(BaseModel):
    """
    Accepts a PyPIConGPU Simulation and runs it

    Manages 2 basic parts:

    - *where* which data is stored (various ``..._dir`` options)
    - *what* is done (generate, build, run)

    Where:

    - run_dir: directory where data for an execution is stored
    - setup_dir: directory where data is generated to and the simulation
      executable is built

    These dirs are either copied from params or guessed.
    See __init__() for a detailed description.

    The initialization of the dirs happens only once (!) inside __init__().
    Any changes performed after that will be accepted and might lead to broken
    builds.

    What:

    - generate(): create a setup (directory) which represents the parameters
      given
    - build(): run pic-build
    - run(): run tbg

    Typically these can only be performed in that order, and each once.
    Whether a step can be started is determined by some sanity checks:
    Are the inputs (e.g. the setup dir, the ``.build`` dir) ready,
    and is the output location empty (e.g. the run dir).
    **If those sanity checks pass, the respective process is launched.**
    If this launched program (e.g. pic-build) fails,
    the process output (stdout & stderr) is printed.
    While a process is running, all output is silenced
    (and collected into an internal buffer).
    """

    template_dir: Annotated[Sequence[Path], AfterValidator(lambda t: tuple(p.absolute() for p in t))] = (tpath(),)
    setup_dir: Annotated[Path, AfterValidator(Path.absolute)] = Field(
        default_factory=lambda: Path(get_tmpdir_with_name("setup")).absolute()
    )
    run_dir: Annotated[Path, AfterValidator(Path.absolute)] = Field(
        default_factory=lambda: Path(get_tmpdir_with_name("run")).absolute()
    )
    sim: Annotated[Simulation, BeforeValidator(lambda s: alt(lambda: s.get_as_pypicongpu(), s))]

    def _log_dirs(self):
        """print human-readble list of paths to log"""
        logging.info(" template dir: {}".format(self.template_dir))
        logging.info("    setup dir: {}".format(self.setup_dir))
        logging.info("      run dir: {}".format(self.run_dir))

    def _render_templates(self):
        """
        render the templates in the setup dir into a picongpu input

        Delegates work to Renderer(), see there for details.
        """
        logging.info("rendering templates...")
        # This is kind of a dirty hack:
        self.sim.spread_directory_information(self.setup_dir)
        # check 1 (implicit): according to schema?
        context = self.sim.get_rendering_context()
        # check 2: structure suitable for renderer?
        Renderer.check_rendering_context(context)
        # dump checked context
        self.store_metadata(context, filename="pypicongpu_rendering_context.json")
        # preprocess (floats to str, add _special properties, ...)
        Renderer.render_directory(Renderer.get_context_preprocessed(context), str(self.setup_dir))

    @property
    def metadata_path(self):
        return self.setup_dir / "metadata"

    @property
    def workflow_dir_path(self):
        return self.setup_dir / "workflow"

    @property
    def profile_path(self):
        return self.workflow_dir_path / "script" / "picongpu.profile"

    @property
    def build_script_path(self):
        return self.workflow_dir_path / "scripts" / "build.sh"

    @property
    def run_script_path(self):
        return self.workflow_dir_path / "scripts" / "run.sh"

    @property
    def workflow_definition_path(self):
        return self.workflow_dir_path / "workflow.cwl"

    @property
    def workflow_input_path(self):
        return self.workflow_dir_path / "input.yaml"

    @property
    def build_step_path(self):
        return self.workflow_dir_path / "steps" / "build.cwl"

    @property
    def run_step_path(self):
        return self.workflow_dir_path / "steps" / "run.cwl"

    def generate_profile(self):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        generate_bare_profile(self.profile_path)

    def generate_build_command(self, *args, rc_params=rc_params, **flags):
        flags = {"j": 4} | flags
        self.build_script_path.parent.mkdir(parents=True, exist_ok=True)
        with self.build_script_path.open("w") as script:
            script.write(
                script_content_with(
                    PicBuildCmdline(args=args, flags=flags).model_dump(),
                    rc_params=rc_params,
                    working_dir=self.setup_dir,
                )
            )
            script.flush()

    def generate_run_command(self, *args, rc_params=rc_params, **flags):
        flags = dict(s="bash", c="etc/picongpu/N.cfg", t="$TBG_TPLFILE") | flags
        args = args or (str(self.run_dir),)
        self.run_script_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_script_path.open("w") as script:
            script.write(
                script_content_with(
                    [
                        f'export PIC_PROFILE="{str(self.profile_path)}"',
                        TbgCmdLine(args=args, flags=flags).model_dump(),
                    ],
                    rc_params=rc_params,
                    working_dir=self.setup_dir,
                )
            )
            script.flush()

    def generate_workflow_input(self):
        data = {
            "run_script": {
                "class": "File",
                "path": str(self.run_script_path),
                # For some reason, the "location" must also be set.
                # See https://github.com/common-workflow-language/cwltool/issues/828#issuecomment-405820330
                "location": str(self.run_script_path),
            },
            "build_script": {
                "class": "File",
                "path": str(self.build_script_path),
                # For some reason, the "location" must also be set.
                # See https://github.com/common-workflow-language/cwltool/issues/828#issuecomment-405820330
                "location": str(self.build_script_path),
            },
        }
        with (self.workflow_input_path).open("w") as file:
            # Technically, we are writing json here,
            # but yaml is a superset of json, so that's fine.
            json.dump(data, file, indent=4)

    def store_metadata(self, metadata, filename):
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        with (self.metadata_path / filename).open("w") as file:
            json.dump(metadata, file, indent=4)

    def generate(self, printDirToConsole=False, exist_ok=False, **flags):
        """
        generate the picongpu-compatible input files
        """

        if printDirToConsole:
            print(" [" + str(self.setup_dir) + "]")

        if not exist_ok:
            assert not self.setup_dir.is_dir(), (
                "setup directory must not exist before generation -- did you call generate() already?"
            )
        preset = rc_params.preset_dir
        copytree(core.path("etc") / f"picongpu/{preset}", self.setup_dir / f"etc/picongpu/{preset}")
        for path in (core.path("etc") / "picongpu").iterdir():
            if path.is_file():
                copy2(path, self.setup_dir / f"etc/picongpu/{path.name}")

        for t in self.template_dir:
            for src, dst in map(
                lambda f: (t / f, self.setup_dir / f),
                ("etc/picongpu", "bin", "include/picongpu", "lib", "validation", "workflow"),
            ):
                if src.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    copytree(src, dst, dirs_exist_ok=True)

        self._render_templates()

        self.generate_profile()
        self.generate_build_command(**flags)
        self.generate_run_command(**flags)
        self.generate_workflow_input()

        self.store_metadata(self.model_dump(mode="json"), filename="pypicongpu_runner.json")
        self.store_metadata(rc_params.model_dump(mode="json"), filename="rc_params.json")

        self._write_rocrate()

    def _write_rocrate(self):
        rc_params.rocrate_info.add_metadata_to(ROCrate(self.setup_dir, init=True)).metadata.write(self.setup_dir)

    def build(self, **flags):
        """
        build (compile) picongpu-compatible input files
        """
        assert self.setup_dir.is_dir(), (
            "setup directory must exist (and contain generated files) -- did you call generate()?"
        )
        assert not (self.setup_dir / ".build").exists(), (
            "build dir (.build in setup dir) must not exist -- did you call build() already?"
        )
        if not self.build_script_path.exists() or flags:
            self.generate_build_command(**flags)
        with self.workflow_input_path.open("r") as file:
            return WorkflowFactory().make(str(self.build_step_path))(script=json.load(file)["build_script"])

    def run(self, **flags):
        """
        run compiled picongpu simulation
        """
        if not self.run_script_path.exists() or flags:
            self.generate_run_command(str(self.run_dir), **flags)
        with self.workflow_input_path.open("r") as file:
            return WorkflowFactory().make(str(self.workflow_definition_path))(**json.load(file))
