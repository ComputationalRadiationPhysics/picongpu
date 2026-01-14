"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

# make pypicongpu classes accessible for conversion to pypicongpu
import datetime
import logging
import math
import typing
from functools import reduce
from itertools import chain, groupby
from os import PathLike
from pathlib import Path

import picmistandard
import typeguard
from pydantic import BaseModel

from picongpu.picmi.diagnostics.particle_dump import ParticleDump
from picongpu.picmi.diagnostics.field_dump import _FieldDump, NativeFieldDump
from picongpu.picmi.interaction import Synchrotron
from picongpu.picmi.interaction.collision import Collision, CollisionalPhysicsSetup
from picongpu.picmi.layout import AnyLayout
from picongpu.picmi.species_requirements import (
    SimpleDensityOperation,
    SimpleMomentumOperation,
    get_as_pypicongpu,
    resolving_add,
    run_construction,
)
from picongpu.pypicongpu.output.openpmd_plugin import FieldDump as PyPIConGPUFieldDump
from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDPlugin
from picongpu.pypicongpu.species.attribute.momentum import Momentum
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams
from picongpu.pypicongpu.util import unique
from picongpu.pypicongpu.walltime import Walltime

from .. import pypicongpu
from . import constants
from .grid import Cartesian3DGrid
from .interaction import Interaction
from .species import Species


class _DensityImpl(BaseModel):
    layout: AnyLayout
    grid: Cartesian3DGrid
    species: Species

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.species.register_requirements(
            [
                Weighting(),
                SimpleDensityOperation(species=self.species, layout=self.layout, grid=self.grid),
                Momentum(),
                SimpleMomentumOperation(species=self.species),
            ]
        )


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _not_allowed_template_directories(directories: tuple[Path]) -> dict[Path, str]:
    """
    Check the directories and return a path->reason mapping of non-allowed ones.
    """
    return {d: "is not an existing directory" for d in filter(lambda p: not p.is_dir(), directories)}


def _normalise_template_dir(directory: None | PathLike | typing.Iterable[PathLike]) -> tuple[Path]:
    """
    Allow strings, Paths and an iterable thereof and return tuple[Path].
    """
    # The ordering of these recursions matters!
    if directory is None:
        return tuple()

    try:
        directory = (Path(directory),)
    except TypeError:
        try:
            directory = sum(map(_normalise_template_dir, directory), tuple())
        except TypeError:
            pass

    if any(filter(lambda p: not isinstance(p, Path), directory)):
        raise ValueError(
            f"Can't understand {directory=} of {type(directory)=}. Must be one of str, Path or iterable thereof."
        )

    if not_allowed := _not_allowed_template_directories(directory):
        raise ValueError(f"Found {not_allowed=} as values for template directories. These are invalid.")
    return directory


def handled_via_openpmd(diagnostic):
    return isinstance(diagnostic, (ParticleDump, _FieldDump))


def _validate_collisional_physics_setup(interactions):
    # Validation is meant in the pydantic sense of checking correctness AND constructing.
    def by_type(x):
        return (
            "collision"
            if isinstance(x, Collision)
            else ("setup" if isinstance(x, CollisionalPhysicsSetup) else "other")
        )

    types = {key: list(values) for key, values in groupby(sorted(interactions, key=by_type), key=by_type)}

    if "setup" in types:
        if "collision" in types:
            raise ValueError(
                f"If you give a CollisionalPhysicsSetup, you have to subsume all collisions under it. You gave: {types['collision']=} and {types['setup']=}."
            )
        if len(list(types["setup"])) > 1:
            raise ValueError(f"Please, only provide at most one CollisionalPhysicsSetup. You gave {types['setup']=}.")
        # It's fine, interactions is consistent just the way it is.
        return interactions

    if "collision" in types:
        # We've found one or more bare collision flying around in the list,
        # so we've gotta merge them into one setup.
        return list(types["other"]) + [CollisionalPhysicsSetup(collisions=list(types["collision"]))]

    # No collisions whatsoever...
    return interactions


# may not use pydantic since inherits from _DocumentedMetaClass
@typeguard.typechecked
class Simulation(picmistandard.PICMI_Simulation):
    """
    Simulation as defined by PICMI

    please refer to the PICMI documentation for the spec
    https://picmi-standard.github.io/standard/simulation.html
    """

    picongpu_custom_user_input = pypicongpu.util.build_typesafe_property(
        typing.Optional[list[pypicongpu.customuserinput.CustomUserInput]]
    )
    """
    list of custom user input objects

    update using picongpu_add_custom_user_input() or by direct setting
    """

    picongpu_interaction = pypicongpu.util.build_typesafe_property(list[Interaction])
    """Interaction instance containing all particle interactions of the simulation, set to None to have no interactions"""

    picongpu_typical_ppc = pypicongpu.util.build_typesafe_property(typing.Optional[int])
    """
    typical number of particle in a cell in the simulation

    used for normalization of code units

    optional, if set to None, will be set to median ppc of all species ppcs
    """

    picongpu_template_dir = pypicongpu.util.build_typesafe_property(typing.Iterable[Path])
    """directory containing templates to use for generating picongpu setups"""

    picongpu_moving_window_move_point = pypicongpu.util.build_typesafe_property(typing.Optional[float])
    """
    point a light ray reaches in y from the left border until we begin sliding the simulation window with the speed of
    light

    in multiples of the simulation window size

    @attention if moving window is active, one gpu in y direction is reserved for initializing new spaces,
        thereby reducing the simulation window size accordingrelative spot at which to start moving the simulation window
    """

    picongpu_moving_window_stop_iteration = pypicongpu.util.build_typesafe_property(typing.Optional[int])
    """iteration, at which to stop moving the simulation window"""

    picongpu_base_density = pypicongpu.util.build_typesafe_property(typing.Optional[float])
    """value to normalise densities with"""

    picongpu_walltime = pypicongpu.util.build_typesafe_property(typing.Optional[datetime.timedelta])
    """time after which the cluster scheduler will stop the simulation"""

    picongpu_binomial_current_interpolation = pypicongpu.util.build_typesafe_property(bool)
    """switch on a binomial current interpolation"""

    picongpu_distributions = pypicongpu.util.build_typesafe_property(list[_DensityImpl])

    __runner = pypicongpu.util.build_typesafe_property(typing.Optional[pypicongpu.runner.Runner])

    # @todo remove boiler plate constructor argument list once picmistandard reference implementation switches to
    #   pydantic, Brian Marre, 2024
    def __init__(
        self,
        picongpu_template_dir: None | PathLike | typing.Iterable[PathLike] = None,
        picongpu_typical_ppc: typing.Optional[int] = None,
        picongpu_moving_window_move_point: typing.Optional[float] = None,
        picongpu_moving_window_stop_iteration: typing.Optional[int] = None,
        picongpu_interaction: typing.Optional[list[Interaction]] = None,
        picongpu_base_density: typing.Optional[float] = None,
        picongpu_walltime: typing.Optional[datetime.timedelta] = None,
        picongpu_binomial_current_interpolation: bool = False,
        **keyword_arguments,
    ):
        self.picongpu_distributions = []
        self.picongpu_template_dir = _normalise_template_dir(picongpu_template_dir)
        self.picongpu_moving_window_move_point = picongpu_moving_window_move_point
        self.picongpu_moving_window_stop_iteration = picongpu_moving_window_stop_iteration
        self.picongpu_base_density = picongpu_base_density
        self.picongpu_walltime = picongpu_walltime
        self.picongpu_binomial_current_interpolation = picongpu_binomial_current_interpolation
        self.picongpu_custom_user_input = None
        self.__runner = None

        if picongpu_typical_ppc is not None and picongpu_typical_ppc <= 0:
            raise ValueError(f"Typical ppc should be > 0, not {picongpu_typical_ppc=}.")
        self.picongpu_typical_ppc = picongpu_typical_ppc

        self.picongpu_interaction = _validate_collisional_physics_setup(picongpu_interaction or [])

        picmistandard.PICMI_Simulation.__init__(self, **keyword_arguments)

        # additional PICMI stuff checks, @todo move to picmistandard, Brian Marre, 2024
        ## throw if both cfl & delta_t are set
        if (
            self.solver is not None
            and self.solver.method in ["Yee", "Lehe"]
            and isinstance(self.solver.grid, Cartesian3DGrid)
        ):
            self.__yee_compute_cfl_or_delta_t()

    def __yee_compute_cfl_or_delta_t(self) -> None:
        """
        use delta_t or cfl to compute the other

        needs grid parameters for computation
        Only works if method is Yee or Lehe.

        :throw AssertionError: if grid (of solver) is not 3D cartesian grid
        :throw AssertionError: if solver is None
        :throw AssertionError: if solver is not "Yee"
        :throw ValueError: if both cfl & delta_t are set, and they don't match

        Does not check if delta_t could be computed
        from max time steps & max time!!

        Exhibits the following behavior:

        delta_t set, cfl not:
          compute cfl
        delta_t not set, cfl set:
          compute delta_t
        delta_t set, cfl also set:
          check both against each other, raise ValueError if they don't match
        delta_t not set, cfl not set either:
          nop (do nothing)
        """
        assert self.solver is not None
        assert self.solver.method in ["Yee", "Lehe"]
        assert isinstance(self.solver.grid, Cartesian3DGrid)

        delta_x = (
            self.solver.grid.upper_bound[0] - self.solver.grid.lower_bound[0]
        ) / self.solver.grid.number_of_cells[0]
        delta_y = (
            self.solver.grid.upper_bound[1] - self.solver.grid.lower_bound[1]
        ) / self.solver.grid.number_of_cells[1]
        delta_z = (
            self.solver.grid.upper_bound[2] - self.solver.grid.lower_bound[2]
        ) / self.solver.grid.number_of_cells[2]

        if self.time_step_size is not None and self.solver.cfl is not None:
            # both cfl & delta_t given -> check their compatibility
            delta_t_from_cfl = self.solver.cfl / (
                constants.c * math.sqrt(1 / delta_x**2 + 1 / delta_y**2 + 1 / delta_z**2)
            )

            if delta_t_from_cfl != self.time_step_size:
                raise ValueError(
                    "time step size (delta t) does not match CFL "
                    "(Courant-Friedrichs-Lewy) parameter! delta_t: {}; "
                    "expected from CFL: {}".format(self.time_step_size, delta_t_from_cfl)
                )
        else:
            if self.time_step_size is not None:
                # calculate cfl
                self.solver.cfl = self.time_step_size * (
                    constants.c * math.sqrt(1 / delta_x**2 + 1 / delta_y**2 + 1 / delta_z**2)
                )
            elif self.solver.cfl is not None:
                # calculate delta_t
                self.time_step_size = self.solver.cfl / (
                    constants.c * math.sqrt(1 / delta_x**2 + 1 / delta_y**2 + 1 / delta_z**2)
                )

            # if neither delta_t nor cfl are given simply silently pass
            # (might change in the future)

    def write_input_file(
        self,
        file_name: str,
        pypicongpu_simulation: typing.Optional[pypicongpu.simulation.Simulation] = None,
    ) -> None:
        """
        generate input data set for picongpu

        file_name must be path to a not-yet existing directory (will be filled
        by pic-create)
        :param file_name: not yet existing directory
        :param pypicongpu_simulation: manipulated pypicongpu simulation
        """
        if self.__runner is not None:
            logging.warning("runner already initialized, overwriting")

        # if not overwritten generate from current state
        if pypicongpu_simulation is None:
            pypicongpu_simulation = self.get_as_pypicongpu()

        self.__runner = pypicongpu.runner.Runner(pypicongpu_simulation, self.picongpu_template_dir, setup_dir=file_name)
        self.__runner.generate()

    def picongpu_add_custom_user_input(self, custom_user_input: pypicongpu.customuserinput.CustomUserInput):
        """add custom user input to previously stored input"""
        self.picongpu_custom_user_input = (self.picongpu_custom_user_input or []) + [custom_user_input]

    def add_interaction(self, interaction) -> None:
        pypicongpu.util.unsupported(
            "PICMI standard interactions are not supported by PIConGPU, use the picongpu specific Interaction object instead"
        )

    # @todo add refactor once restarts are supported by the Runner, Brian Marre, 2024
    def step(self, nsteps: int = 1):
        if nsteps != self.max_steps:
            raise ValueError(
                "PIConGPU does not support stepwise running. Invoke step() with max_steps (={})".format(self.max_steps)
            )
        self.picongpu_run()

    def _generate_openpmd_plugins(self, diagnostics, num_steps):
        diagnostics = list(diagnostics)
        return [
            OpenPMDPlugin(
                sources=[
                    (
                        diagnostic.period.get_as_pypicongpu(time_step_size=self.time_step_size, num_steps=num_steps),
                        diagnostic.species.get_as_pypicongpu()
                        if isinstance(diagnostic, ParticleDump)
                        else PyPIConGPUFieldDump(
                            name=diagnostic.fieldname,
                            functor=None
                            if isinstance(diagnostic, NativeFieldDump)
                            else diagnostic.functor.get_as_pypicongpu(),
                        ),
                    )
                    for diagnostic in filter(lambda x: x.options == options, diagnostics)
                ],
                config=options,
            )
            for options in unique(map(lambda x: x.options, diagnostics))
        ]

    def _generate_plugins(self, num_steps):
        return [
            entry.get_as_pypicongpu(
                time_step_size=self.time_step_size,
                num_steps=num_steps,
            )
            for entry in self.diagnostics
            if not handled_via_openpmd(entry)
        ] + self._generate_openpmd_plugins(filter(handled_via_openpmd, self.diagnostics), num_steps)

    def _check_compatibility(self):
        pypicongpu.util.unsupported("verbose", self.verbose)
        pypicongpu.util.unsupported("particle shape", self.particle_shape, "linear")
        pypicongpu.util.unsupported("gamma boost", self.gamma_boost)
        if len(self.laser_injection_methods) != self.laser_injection_methods.count(None):
            pypicongpu.util.unsupported("laser injection method", self.laser_injection_methods, [])
        if self.max_steps is None and self.max_time is None:
            raise ValueError("runtime not specified (neither as step count nor max time)")

    def get_as_pypicongpu(self) -> pypicongpu.simulation.Simulation:
        """translate to PyPIConGPU object"""
        self._check_compatibility()

        init_operations = organise_init_operations(
            chain(*(s.get_operation_requirements() for s in sorted(self.species)))
        )

        typical_ppc = (
            self.picongpu_typical_ppc
            if self.picongpu_typical_ppc is not None
            else _mid_window(map(lambda op: op.layout.ppc, filter(lambda op: hasattr(op, "layout"), init_operations)))
        )
        moving_window = (
            None
            if self.picongpu_moving_window_move_point is None
            else pypicongpu.movingwindow.MovingWindow(
                move_point=self.picongpu_moving_window_move_point,
                stop_iteration=self.picongpu_moving_window_stop_iteration,
            )
        )
        walltime = (
            None if self.picongpu_walltime is None else pypicongpu.walltime.Walltime(walltime=self.picongpu_walltime)
        )
        time_steps = self.max_steps if self.max_steps is not None else math.ceil(self.max_time / self.time_step_size)
        # We provide the default as last element and we'll only read the first element:
        synchrotron_params = unique(
            [x.synchrotron_parameters for x in self.picongpu_interaction if isinstance(x, Synchrotron)]
        ) + [SynchrotronParams()]
        if len(synchrotron_params) > 2:
            raise ValueError(
                f"You have configured the Synchrotron extension multiple times with different arguments. This is not allowed! You gave {synchrotron_params[:-1]=}."
            )
        # We provide the default as last element and we'll only read the first element:
        collisions = [x for x in self.picongpu_interaction if isinstance(x, CollisionalPhysicsSetup)] + [
            CollisionalPhysicsSetup()
        ]

        return pypicongpu.simulation.Simulation(
            species=map(get_as_pypicongpu, sorted(self.species)),
            init_operations=init_operations,
            typical_ppc=typical_ppc,
            delta_t_si=self.time_step_size,
            solver=self.solver.get_as_pypicongpu(),
            customuserinput=self.picongpu_custom_user_input,
            grid=self.solver.grid.get_as_pypicongpu(),
            binomial_current_interpolation=self.picongpu_binomial_current_interpolation,
            moving_window=moving_window,
            walltime=walltime or Walltime(walltime=datetime.timedelta(hours=1)),
            time_steps=time_steps,
            laser=[ll.get_as_pypicongpu() for ll in self.lasers] or None,
            output=self._generate_plugins(time_steps),
            base_density=self._get_base_density(),
            synchrotron_params=synchrotron_params[0],
            collisional_physics=collisions[0].get_as_pypicongpu(),
        )

    def _get_base_density(self) -> float:
        return self.picongpu_base_density or 1.0e25

    def picongpu_run(self) -> None:
        """build and run PIConGPU simulation"""
        runner = self.picongpu_get_runner()
        runner.generate()
        runner.build()
        runner.run()

    def picongpu_get_runner(self) -> pypicongpu.runner.Runner:
        if self.__runner is None:
            self.__runner = pypicongpu.runner.Runner(self.get_as_pypicongpu(), self.picongpu_template_dir)
        return self.__runner

    def _picongpu_add_species(self, species, layout):
        self.species.append(species)
        self.layouts.append(layout)
        if species.density_scale is not None and (layout is None and species.initial_distribution is None):
            raise ValueError("layout and initial distribution must be set to use density scale")
        if layout is not None and species.initial_distribution is None:
            raise ValueError(
                f"An initial distribution needs a layout. You've given {layout=} but {species.initial_distribution=}."
            )
        if species.initial_distribution is not None:
            self.picongpu_distributions.append(_DensityImpl(species=species, layout=layout, grid=self.solver.grid))

    def add_species(self, *args, **kwargs):
        return self._picongpu_add_species(*args, **kwargs)


def organise_init_operations(operations):
    cleaned = []
    for op in operations:
        cleaned = resolving_add(op, cleaned)
    return [run_construction(op) for op in cleaned]


def _mid_window(iterable):
    """Compute the integer in the middle between min(iterable), max(iterable), return 1 if empty."""
    iterable = iter(iterable)

    try:
        start = next(iterable)
    except StopIteration:
        return 1

    mi, ma = reduce(lambda lhs, rhs: (min(lhs[0], rhs), max(lhs[1], rhs)), iterable, (start, start))
    return int((ma - mi) // 2 + mi)
