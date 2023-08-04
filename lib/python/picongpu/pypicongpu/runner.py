"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from .simulation import Simulation
from . import util
from .rendering import Renderer

from os import path, environ, chdir
from typeguard import typechecked
from functools import reduce
import tempfile
import subprocess
import logging
import typing
import re
import datetime
import pathlib
import json


def runArgs(name, args):
    assert list(filter(lambda x: x is None, args)) == [], \
        "arguments must not be None!"
    logging.info('running {}...'.format(name))
    logging.debug('command for {}: {}'.format(name, ' '.join(args)))
    proc = subprocess.run(args,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    logging.info('{} done, returned {}'.format(name, proc.returncode))

    if 0 != proc.returncode:
        logging.error(
            '>>>>>>> Command failed (output below): {}\n{}'.
            format(' '.join(proc.args), proc.stdout.decode()))
        logging.error(
            '>>>>>>> Command failed (output above): {}'.
            format(' '.join(proc.args)))
        raise RuntimeError('subprocess failed')


def get_tmpdir_with_name(name, parent: str = None):
    """
    returns a not existing temporary directory path,
    which contains the given name
    :param name: part of the newly created directory name
    :param parent: if given: create the tmpdir there
    :return: not existing path to directory
    """
    assert re.match("^[0-9a-zA-Z._-]*$", name), \
        "generated dir name may only contain a-zA-Z0-9._-"

    # Note: Do *not* use isotime here,
    # the colon (:) seems to screw with pic-build.
    # Also, don't use "T" as separator, to not get confuse with isotime format.
    prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # important note on how these directories are created:
    # the TemporaryDirectory() object *creates* the directory,
    # immediately goes out of scope and deletes the dir again
    # -> we are left with purely a name
    dir_name = None
    with tempfile.TemporaryDirectory(
            prefix="pypicongpu-{}-{}-".format(prefix, name),
            dir=parent) as tmpdir:
        # dir now exists
        dir_name = tmpdir
    assert not path.exists(dir_name), \
        "freshly generated tmp dir name should not exist (anymore)"
    return dir_name


@typechecked
class Runner():
    """
    Accepts a PyPIConGPU Simulation and runs it

    Manages 2 basic parts:

    - *where* which data is stored (various ``..._dir`` options)
    - *what* is done (generate, build, run)

    Where:

    - scratch_dir: (optional) directory where many simulation results can be
      stored
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

    SCRATCH_ENV_NAME = "SCRATCH"
    """name of the environment variable where the scratch dir defaults to"""

    setup_dir = util.build_typesafe_property(str)
    """
    directory containing the experiment setup (scenario)

    pic-build is called here
    """

    scratch_dir = util.build_typesafe_property(typing.Optional[str])
    """directory where run directories can be store"""

    run_dir = util.build_typesafe_property(str)
    """directory where run results (and a copy of the inputs) will be stored"""

    sim = util.build_typesafe_property(Simulation)
    """the picongpu simulation to be run"""

    __valid_path_re = re.compile("^[a-zA-Z0-9/._-]+$")
    """
    regex that matches a valid path. Note: allows *less* characters than the OS
    """

    def __init__(self, sim,
                 pypicongpu_template_dir: typing.Optional[str] = None,
                 scratch_dir: typing.Optional[str] = None,
                 setup_dir: typing.Optional[str] = None,
                 run_dir: typing.Optional[str] = None):
        """
        initialize self using simulation and (maybe) given paths

        - A *scratch dir* is a directory of semi-permanent storage,
          where results of runs are collected.
          Typically it holds many run directories.
        - A *setup dir* describes one experiment (scenario).
          You can call pic-build there.
        - A *run dir* holds data of a single PIConGPU run:
          All input including the built binary in input/,
          the results in a directory simOutput/
          (and maybe additional files, e.g. tbg/).

        If not given the paths are guessed as follows:

        - If not given the scratch dir is the value of the environment variable
          $SCRATCH
        - If not given the run dir is derived from the scratch dir.
        - If neither run nor scratch dir are given,
          a warning is printed that the run is considered temporary and
          the run dir created as temporary directory.
          The scratch dir is left at None.
          TODO in this case multi-device support is disabled.
        - If the setup directory is not given a
          temporary directory will be used
          (as the setup itself and its built results will be copied to
          the run dir's subdir input/ anyways).

        After the paths have been set the following applies:

        - setup_dir does not exist (will be filled by generate() and build())
        - run_dir does not exist (will be filled by run()).
          If scratch_dir is given, run_dir is a child of it.

        Note: Catch type of sim not via typeguard b/c that would
        require circular imports. Type of sim is checked manually.

        :param sim: simulation to be built
        :param pypicongpu_template_dir: path to pypicongpu template to be
        copied, guessed by default
        :param scratch_dir: directory where results can be stored
        :param setup_dir: not-yet existing directory where build files
        (.params, built binary, etc.) will be stored
        :param run_dir: not-yet existing directory where results will be stored
        """

        # note: only import here to prevent circular imports
        from .. import picmi

        if isinstance(sim, Simulation):
            self.sim = sim
        elif isinstance(sim, picmi.Simulation):
            self.sim = sim.get_as_pypicongpu()
        else:
            raise TypeError(
                "sim must be pypicongpu simulation or picmi simulation, "
                "got: {}".format(type(sim)))

        # use helper to perform various checks
        # note that the order matters: run_dir depends on scratch_dir
        self.__helper_set_pypicongpu_template_dir(pypicongpu_template_dir)
        self.__helper_set_scratch_dir(scratch_dir)
        self.__helper_set_setup_dir(setup_dir)
        self.__helper_set_run_dir(run_dir)

        # dump used paths for diagnostics
        self.__log_dirs()

        # collision checks
        assert self.scratch_dir != self.setup_dir, \
            "scratch dir must not be equal to the setup dir"
        assert self.setup_dir != self.run_dir, \
            "setup dir must not be equal to the run dir"
        assert self.run_dir != self.scratch_dir, \
            "run dir must not be equal to the scratch dir"

    def __helper_set_setup_dir(self, setup_dir: typing.Optional[str]) -> None:
        """sets the setup dir according to description in __init__()"""
        assert setup_dir is None or self.__valid_path_re.match(setup_dir), \
            "setup dir contains invalid characters"
        # setup dir (given or /tmp)
        if setup_dir is not None:
            self.setup_dir = path.abspath(setup_dir)
        else:
            # just place in /tmp, will (1) be needed on local machine only
            # and (2) be copied to run_dir/input by tbg
            self.setup_dir = get_tmpdir_with_name("setup")
        assert not path.isdir(self.setup_dir), "setup dir must NOT exist yet"

    def __helper_set_pypicongpu_template_dir(
            self, pypicongpu_template_dir: typing.Optional[str]) -> None:
        """sets the pypicongpu template dir as described in __init__()"""
        # guess template
        # store in private var, because people should not mess with it
        if pypicongpu_template_dir is None:
            # find source of pypicongpu repo,
            # from there derive template location
            self.__pypicongpu_template_dir = \
                path.join(reduce(lambda x, f: f(x),
                                 [path.dirname] * 5,
                                 __file__),
                          "share/picongpu/pypicongpu/template")
        else:
            self.__pypicongpu_template_dir = \
                path.abspath(pypicongpu_template_dir)
        assert path.isdir(self.__pypicongpu_template_dir), \
            "template directory must exist"

    def __helper_set_scratch_dir(
            self, scratch_dir: typing.Optional[str]) -> None:
        """sets the scratch dir according to description in __init__()"""
        assert scratch_dir is None or self.__valid_path_re.match(
            scratch_dir), "scratch dir contains invalid characters"
        # scratch dir (given, or environment, else None)
        if scratch_dir is not None:
            self.scratch_dir = path.abspath(scratch_dir)
        else:
            # try to retrieve from environment var
            if self.SCRATCH_ENV_NAME in environ:
                logging.info("loading scratch directory (implicitly) "
                             "from environment var ${}".
                             format(self.SCRATCH_ENV_NAME))
                self.scratch_dir = path.abspath(environ[self.SCRATCH_ENV_NAME])
            else:
                self.scratch_dir = None

        if self.scratch_dir is not None and \
           self.scratch_dir.startswith(str(pathlib.Path.home())):
            logging.warning("You specified your scratch directory to be inside"
                            " your $HOME. THIS IS NOT ACCEPTABLE ON HPC!")
        assert self.scratch_dir is None or path.isdir(self.scratch_dir), \
            "scratch directory must exist"

    def __helper_set_run_dir(self, run_dir: typing.Optional[str]) -> None:
        """sets the run dir according to description in __init__()"""
        assert run_dir is None or self.__valid_path_re.match(run_dir), \
            "run dir contains invalid characters"
        # run dir
        # (given or placed in scratch dir or put into /tmp with warning)
        if run_dir is not None:
            self.run_dir = path.abspath(run_dir)
        else:
            if self.scratch_dir is not None:
                self.run_dir = get_tmpdir_with_name("run", self.scratch_dir)
            else:
                # note: do not print warning yet,
                # because maybe the user will never use this run dir
                self.run_dir = get_tmpdir_with_name("run")
        assert not path.isdir(self.run_dir), "run dir must NOT exist yet"

    def __params_file(self):
        return path.join(self.setup_dir,
                         "include/picongpu/param/pypicongpu.param")

    def __cfg_file(self):
        return path.join(self.setup_dir, "etc/picongpu/pypicongpu.cfg")

    def __log_dirs(self):
        """print human-readble list of paths to log"""
        logging.info(" template dir: {}"
                     .format(self.__pypicongpu_template_dir))
        logging.info("    setup dir: {}".format(self.setup_dir))
        logging.info("      run dir: {}".format(self.run_dir))
        logging.info("  params file: {}".format(self.__params_file()))
        logging.info("     cfg file: {}".format(self.__cfg_file()))

    def __copy_template(self):
        """copy template files to be built from"""
        runArgs("add template",
                ["pic-create", "--force", self.__pypicongpu_template_dir,
                 self.setup_dir])

    def __render_templates(self):
        """
        render the templates in the setup dir into a picongpu input

        Delegates work to Renderer(), see there for details.
        """
        logging.info("rendering templates...")
        # check 1 (implicit): according to schema?
        context = self.sim.get_rendering_context()
        # check 2: structure suitable for renderer?
        Renderer.check_rendering_context(context)
        # dump checked context
        with open("{}/pypicongpu.json".format(self.setup_dir), "w") as file:
            json.dump(context, file, indent=4)

        # preprocess (floats to str, add _special properties, ...)
        preprocessed_context = Renderer.get_context_preprocessed(context)

        Renderer.render_directory(preprocessed_context, self.setup_dir)

    def __build(self):
        """launch build of PIConGPU"""
        chdir(self.setup_dir)
        runArgs("pic-build", ["pic-build"])

    def __run(self):
        """
        execute PIConGPU

        this uses the N.cfg provided by the template,
        therefore will not work with any other configuration
        TODO multi-device support
        """
        chdir(self.setup_dir)
        runArgs("PIConGPU",
                "tbg -s bash -c etc/picongpu/N.cfg -t "
                "etc/picongpu/bash/mpiexec.tpl".split(" ") + [self.run_dir])

    def generate(self, printDirToConsole=False):
        """
        generate the picongpu-compatible input files
        """

        if printDirToConsole:
            print(" [" + str(self.setup_dir) + "]")

        assert not path.isdir(self.setup_dir), \
            "setup directory must not exist before generation -- "\
            "did you call generate() already?"
        self.__copy_template()
        self.__render_templates()

    def build(self):
        """
        build (compile) picongpu-compatible input files
        """
        assert path.isdir(self.setup_dir), \
            "setup directory must exist (and contain generated files) -- "\
            "did you call generate()?"
        assert not path.isdir(path.join(self.setup_dir, ".build")), \
            "build dir (.build in setup dir) must not exist -- "\
            "did you call build() already?"
        self.__build()

    def run(self):
        """
        run compiled picongpu simulation
        """
        assert path.isdir(path.join(self.setup_dir, ".build")), \
            "build dir (.build in setup dir) must exist -- "\
            "did you call build()?"
        assert not path.isdir(self.run_dir), \
            "run dir must not exist yet -- did you call run() already?"

        if self.run_dir.startswith(path.abspath(tempfile.gettempdir())):
            logging.warning(
                "run dir is inside the temporary directory. "
                "THE SIMULATION RESULTS ARE NOT ON PERMANENT STORAGE!")
            # TODO: maybe note that multi-device support is disabled

        self.__run()
