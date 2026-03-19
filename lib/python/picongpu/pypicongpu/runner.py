"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import datetime
import json
import logging
import subprocess
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from os import chmod
from pathlib import Path
from typing import Annotated, Sequence

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, PrivateAttr
from rocrate.rocrate import ROCrate

from picongpu import core, rc_params
from picongpu.templates import path as tpath

from .rendering import Renderer
from .simulation import Simulation
from .util import alt


def runArgs(name, args):
    assert list(filter(lambda x: x is None, args)) == [], "arguments must not be None!"
    logging.info("running {}...".format(name))
    logging.debug("command for {}: {}".format(name, " ".join(args)))
    proc = subprocess.run(args, capture_output=True)
    logging.info("{} done, returned {}".format(name, proc.returncode))
    logging.debug(f"command for {name}: {' '.join(args)}\n{proc.stdout.decode()=}")
    logging.debug(f"command for {name}: {' '.join(args)}\n{proc.stderr.decode()=}")

    if 0 != proc.returncode:
        logging.error(">>>>>>> Command failed (output below): {}\n{}".format(" ".join(proc.args), proc.stdout.decode()))
        logging.error(">>>>>>> Command failed (output above): {}".format(" ".join(proc.args)))
        raise RuntimeError("subprocess failed")

    return proc


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


def run_commands(commands, name=None, script_path=None, rc_params=rc_params, working_dir=None):
    name = name or "unnamed_task"
    if script_path is not None:
        script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w") if script_path is not None else tempfile.NamedTemporaryFile(mode="w") as script:
        script.write(script_content_with(commands, rc_params=rc_params, working_dir=working_dir))
        script.flush()
        return runArgs(name, [*rc_params.shebang[len("#!") :].split(" "), str(script.name)])


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

    _rocrate: ROCrate = PrivateAttr(default_factory=lambda: ROCrate(version="1.2"))
    _temporary_metadata_storage_value: Path | None = PrivateAttr(None)

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
        with tempfile.TemporaryDirectory() as d:
            self._rocrate.write(d)
            rendered_directory = Renderer.render_directory(Renderer.get_context_preprocessed(context), d)

        # The top-level `./` id is kind of special in an RO-Crate,
        # so just `add_tree(..., './')` doesn't quite work as expected.
        for p in rendered_directory.iterdir():
            if p.is_file():
                self._rocrate.add_file(p, p.name)
            else:
                self._rocrate.add_tree(p, p.name)

    @property
    def metadata_path(self):
        return self.setup_dir / "metadata"

    @property
    def profile_path(self):
        return self.setup_dir / "commands" / "picongpu.profile"

    @property
    def build_command_path(self):
        return self.setup_dir / "commands" / "build.sh"

    @property
    def run_command_path(self):
        return self.setup_dir / "commands" / "run.sh"

    def generate_build_command(self, *args, rc_params=rc_params, **flags):
        with tempfile.NamedTemporaryFile("w", delete=False, delete_on_close=False) as script:
            script.write(
                script_content_with(
                    " ".join(["pic-build", *map(lambda f: f"-{f[0]} {f[1]}", flags.items()), *args]),
                    rc_params=rc_params,
                    working_dir=self.setup_dir,
                )
            )
            script.flush()
            path = Path(script.name).absolute()
        chmod(path, 777)
        return path

    def generate_run_command(self, *args, rc_params=rc_params, **flags):
        with tempfile.NamedTemporaryFile("w", delete=False, delete_on_close=False) as script:
            script.write(
                script_content_with(
                    [
                        f'export PIC_PROFILE="{str(self.profile_path)}"',
                        " ".join(["tbg", *map(lambda f: f"-{f[0]} {f[1]}", flags.items()), *args]),
                    ],
                    rc_params=rc_params,
                    working_dir=self.setup_dir,
                )
            )
            script.flush()
            path = Path(script.name).absolute()
        chmod(path, 777)
        return path

    @property
    def _temporary_metadata_storage(self):
        if self._temporary_metadata_storage_value is None:
            self._temporary_metadata_storage_value = Path(tempfile.TemporaryDirectory(delete=False).name).absolute()
        return self._temporary_metadata_storage_value

    def store_metadata(self, metadata, filename):
        with (self._temporary_metadata_storage / filename).open("w") as file:
            json.dump(metadata, file, indent=4)
            file.flush()
            self._rocrate.add_tree(Path(file.name).parent, "metadata")

    def generate(self, printDirToConsole=False):
        """
        generate the picongpu-compatible input files
        """

        if printDirToConsole:
            print(" [" + str(self.setup_dir) + "]")

        assert not self.setup_dir.is_dir(), (
            "setup directory must not exist before generation -- did you call generate() already?"
        )
        preset = rc_params.preset_dir
        self._rocrate.add_tree(core.path("etc") / f"picongpu/{preset}", f"etc/picongpu/{preset}")
        for path in (core.path("etc") / "picongpu").iterdir():
            if path.is_file():
                self._rocrate.add_file(path, f"etc/picongpu/{path.name}")

        for t in self.template_dir:
            for dst in ("etc/picongpu", "bin", "include/picongpu", "lib", "validation"):
                if (src := t / dst).is_dir():
                    self._rocrate.add_tree(src, dst)
            if (wf_file := t / "workflow.cwl").is_file():
                self._rocrate.add_workflow(wf_file, "workflow.cwl")

        self._rocrate.add_file(generate_bare_profile(), "commands/picongpu.profile")
        self._rocrate.add_file(self.generate_build_command(j=4), "commands/build.sh")
        self._rocrate.add_file(
            self.generate_run_command(str(self.run_dir), s="bash", c="etc/picongpu/N.cfg", t="$TBG_TPLFILE"),
            "commands/run.sh",
        )

        self.store_metadata(self.model_dump(mode="json"), filename="pypicongpu_runner.json")
        self.store_metadata(rc_params.model_dump(mode="json"), filename="rc_params.json")

        self._render_templates()
        self._rocrate.write(self.setup_dir)

    def build(self):
        """
        build (compile) picongpu-compatible input files
        """
        assert self.setup_dir.is_dir(), (
            "setup directory must exist (and contain generated files) -- did you call generate()?"
        )
        assert not (self.setup_dir / ".build").exists(), (
            "build dir (.build in setup dir) must not exist -- did you call build() already?"
        )
        if not self.build_command_path.exists():
            self.generate_build_command(j=4)
        return runArgs("build", [*rc_params.shebang[len("#!") :].split(" "), str(self.build_command_path)])

    def run(self):
        """
        run compiled picongpu simulation
        """
        assert (self.setup_dir / ".build").is_dir(), (
            "build dir (.build in setup dir) must exist -- did you call build()?"
        )
        assert not self.run_dir.exists(), "run dir must not exist yet -- did you call run() already?"
        if not self.run_command_path.exists():
            self.generate_run_command(str(self.run_dir), s="bash", c="etc/picongpu/N.cfg", t="$TBG_TPLFILE")
        return runArgs("run", [*rc_params.shebang[len("#!") :].split(" "), str(self.run_command_path)])
