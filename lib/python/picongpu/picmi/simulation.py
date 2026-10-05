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
from functools import reduce
from itertools import chain, groupby
from os import PathLike
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import picmistandard
from pydantic import (
    AfterValidator,
    BeforeValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from sympy import Symbol

from picongpu import pypicongpu, templates
from picongpu.picmi import constants
from picongpu.picmi.diagnostics.field_dump import NativeFieldDump, _FieldDump
from picongpu.picmi.diagnostics.particle_dump import ParticleDump
from picongpu.picmi.diagnostics.phase_space import PhaseSpace
from picongpu.picmi.distribution.AnalyticDistribution import AnalyticDistribution
from picongpu.picmi.grid import Cartesian2DGrid, Cartesian3DGrid, AnyGrid
from picongpu.picmi.interaction import Interaction, Synchrotron
from picongpu.picmi.interaction.collision import Collision, CollisionalPhysicsSetup
from picongpu.picmi.layout import AnyLayout
from picongpu.picmi.species import _STANDARD_SHAPES, Species
from picongpu.picmi.species_requirements import (
    SimpleDensityOperation,
    SimpleMomentumOperation,
    get_as_pypicongpu,
    resolving_add,
    run_construction,
)
from picongpu.picmi.memory_config import MemoryConfig
from picongpu.picmi.precision_config import PrecisionConfig
from picongpu.pypicongpu.output.openpmd_plugin import FieldDump as PyPIConGPUFieldDump
from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDPlugin
from picongpu.pypicongpu.runner import Runner
from picongpu.pypicongpu.species.attribute.momentum import Momentum
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams
from picongpu.pypicongpu.util import UnpackChain, unique
from picongpu.pypicongpu.walltime import Walltime


class _DensityImpl(BaseModel):
    layout: AnyLayout
    grid: AnyGrid
    species: Species

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


def _normalise_template_dir(directory: None | PathLike | Iterable[PathLike]) -> tuple[Path]:
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

    if not isinstance(directory, (tuple, list)) or any(filter(lambda p: not isinstance(p, Path), directory)):
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
        return list(types.get("other", [])) + [CollisionalPhysicsSetup(collisions=list(types["collision"]))]

    # No collisions whatsoever...
    return interactions


# may not use pydantic since inherits from _DocumentedMetaClass
class Simulation(picmistandard.PICMI_Simulation):
    """
    Simulation as defined by PICMI

    please refer to the PICMI documentation for the spec
    https://picmi-standard.github.io/standard/simulation.html
    """

    # Override the standard's particle_shape (default "linear") to default to None:
    # an unset Simulation-level shape lets each species fall back to the PIConGPU
    # default ('quadratic'/TSC), while a set value is inherited by species that
    # don't specify their own shape. The accepted values match Species.particle_shape
    # (the PICMI-standard names plus PIConGPU 'other:' extensions).
    particle_shape: str | None = Field(
        default=None,
        description="Default particle shape for species added to this simulation. "
        "One of 'NGP', 'linear', 'quadratic', 'cubic' or a PIConGPU 'other:' extension "
        "(unlike the PICMI standard, integer interpolation orders are not accepted). "
        "Species without their own particle_shape inherit this value; if it is unset "
        "they fall back to the PIConGPU default 'quadratic' (TSC).",
    )

    @field_validator("particle_shape")
    @classmethod
    def _validate_particle_shape(cls, value):
        if value is not None and value not in _STANDARD_SHAPES and not value.startswith("other:"):
            raise ValueError(
                f"Unsupported particle shape {value!r}. Must be one of "
                f"{', '.join(_STANDARD_SHAPES)} or be prefixed with 'other:'."
            )
        return value

    # Excluded from model dumps because it is passed through to the pypicongpu
    # Simulation as-is (single owner is the pypicongpu model) and because
    # CustomUserInput.rendering_context may hold arbitrary, non-serializable user data.
    picongpu_custom_user_input: list[pypicongpu.customuserinput.CustomUserInput] | None = Field(
        default=None, exclude=True
    )
    """
    list of custom user input objects

    update using picongpu_add_custom_user_input() or by direct setting
    """

    picongpu_interaction: Annotated[list[Interaction], BeforeValidator(_validate_collisional_physics_setup)] = Field(
        default_factory=list
    )
    """Interaction instance containing all particle interactions of the simulation, set to None to have no interactions"""

    def _validate_typical_ppc(value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError(f"Typical ppc should be > 0, not {value=}.")
        return value

    picongpu_typical_ppc: Annotated[int | None, AfterValidator(_validate_typical_ppc)] = Field(default=None)
    """
    typical number of particle in a cell in the simulation

    used for normalization of code units

    optional, if set to None, will be set to the integer midpoint between the
    smallest and largest per-layout ppc of the initialized species
    """

    picongpu_template_dir: Annotated[tuple[Path, ...], BeforeValidator(_normalise_template_dir)] = Field(default=())
    """directory containing templates to use for generating picongpu setups"""

    picongpu_moving_window_move_point: float | None = Field(default=None)
    """
    point a light ray reaches in y from the left border until we begin sliding the simulation window with the speed of
    light

    in multiples of the simulation window size

    @attention if moving window is active, one gpu in y direction is reserved for initializing new spaces,
        thereby reducing the simulation window size accordingrelative spot at which to start moving the simulation window
    """

    picongpu_moving_window_stop_iteration: int | None = Field(default=None)
    """iteration, at which to stop moving the simulation window"""

    picongpu_base_density: float | None = Field(default=None)
    """value to normalise densities with"""

    picongpu_precision: Literal[32, 64] = Field(default=32)
    """
    floating point precision of the simulation core (see ``precision.param``)

    32 (single precision, default) or 64 (double precision). Controls the
    ``precisionPIConGPU`` namespace in the generated ``precision.param``.
    """

    picongpu_precision_config: PrecisionConfig = Field(default_factory=PrecisionConfig)
    """
    per-namespace precision overrides (sqrt/exp/trig) rendered into ``precision.param``
    (see ``PrecisionConfig``).

    ``"core"`` (default) aliases the core ``precisionPIConGPU`` precision; ``32``/``64``
    force ``precision32Bit``/``precision64Bit`` respectively.
    """

    picongpu_memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    """
    memory / exchange-buffer knobs rendered into ``memory.param`` (see ``MemoryConfig``).
    """

    picongpu_walltime: datetime.timedelta | None = Field(default=None)
    """time after which the cluster scheduler will stop the simulation"""

    picongpu_distributions: list[_DensityImpl] = Field(default_factory=list)

    _runner: Runner | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Solvers that impose a CFL stability limit and therefore participate in the
    # delta_t / cfl cross-check below. "other:None" is deliberately excluded: it
    # disables the vacuum field update (and the J->E coupling), so it has no CFL
    # limit at all (the C++ CFLChecker returns infinity) and any delta_t is legal.
    _CFL_GATED_SOLVER_METHODS = ("Yee", "Lehe", "CKC", "other:ArbitraryOrderFDTD")

    @model_validator(mode="after")
    def _post_init(self):
        # cross-check cfl against delta_t, deriving whichever is missing; the
        # "other:None" solver has no CFL limit and is skipped entirely
        if (
            self.solver is not None
            and self.solver.method in self._CFL_GATED_SOLVER_METHODS
            and isinstance(self.solver.grid, (Cartesian3DGrid, Cartesian2DGrid))
        ):
            self._compute_cfl_or_delta_t()
        return self

    def _compute_cfl_or_delta_t(self) -> None:
        """
        use delta_t or cfl to compute the other

        needs grid parameters for computation
        Only works for solvers that impose a CFL limit (Yee, Lehe, CKC and
        other:ArbitraryOrderFDTD); "other:None" has no CFL limit and is skipped.

        :throw AssertionError: if grid (of solver) is not a cartesian grid
        :throw AssertionError: if solver is None
        :throw AssertionError: if solver does not impose a CFL limit
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
        assert self.solver.method in self._CFL_GATED_SOLVER_METHODS
        assert isinstance(self.solver.grid, (Cartesian3DGrid, Cartesian2DGrid))

        # The CFL factor is sqrt(sum over the spatial dimensions of 1/cell_size^2).
        # In 2D the z term is dropped, so a square 2D grid yields a factor of sqrt(2)
        # (not sqrt(3) as in 3D).
        grid = self.solver.grid
        cell_size = [
            (grid.upper_bound[i] - grid.lower_bound[i]) / grid.number_of_cells[i]
            for i in range(grid.number_of_dimensions)
        ]
        cfl_factor = math.sqrt(sum(1.0 / cs**2 for cs in cell_size))

        if self.solver.method in ("Yee", "Lehe"):
            # Legacy second-order FDTD: cfl = delta_t * c * sqrt(sum 1/dx^2).
            # Kept as the original arithmetic (cfl_factor is the dimension-aware
            # version and equals the 3D formula for a 3D grid) so Yee/Lehe results
            # are bit-for-bit unchanged.
            cfl_scale = constants.c * cfl_factor

            def _delta_t_from_cfl(cfl):
                return cfl / cfl_scale

            def _cfl_from_delta_t(delta_t):
                return delta_t * cfl_scale
        else:
            # CKC (min cell) and other:ArbitraryOrderFDTD (Yee term / weight-sum):
            # cfl = c * delta_t / max_c_dt, where max_c_dt is the solver's CFL
            # limit on c * delta_t. The solver owns that restriction, and passing
            # only the spatial cell sizes makes it dimension-aware (2D drops z).
            max_c_dt = self.solver._cfl_max_cdt(*cell_size)
            assert max_c_dt is not None, "solver does not impose a CFL limit"

            def _delta_t_from_cfl(cfl):
                return cfl * max_c_dt / constants.c

            def _cfl_from_delta_t(delta_t):
                return delta_t * constants.c / max_c_dt

        if self.time_step_size is not None and self.solver.cfl is not None:
            # both cfl & delta_t given -> check their compatibility
            delta_t_from_cfl = _delta_t_from_cfl(self.solver.cfl)
            if delta_t_from_cfl != self.time_step_size:
                raise ValueError(
                    "time step size (delta t) does not match CFL "
                    "(Courant-Friedrichs-Lewy) parameter! delta_t: {}; "
                    "expected from CFL: {}".format(self.time_step_size, delta_t_from_cfl)
                )
        else:
            if self.time_step_size is not None:
                # calculate cfl
                self.solver.cfl = _cfl_from_delta_t(self.time_step_size)
            elif self.solver.cfl is not None:
                # calculate delta_t
                self.time_step_size = _delta_t_from_cfl(self.solver.cfl)

            # if neither delta_t nor cfl are given simply silently pass
            # (might change in the future)

    def write_input_file(self, file_name: str | Path, exist_ok=False, **flags) -> None:
        """
        generate input data set for picongpu

        file_name must be path to a not-yet existing directory (will be filled
        by pic-create)
        :param file_name: not yet existing directory
        :param pypicongpu_simulation: manipulated pypicongpu simulation
        """
        if self._runner is not None:
            logging.warning("runner already initialized, overwriting")

        self._runner = Runner(
            sim=self, template_dir=self.picongpu_template_dir or (templates.path(),), setup_dir=Path(file_name)
        )
        self._runner.generate(exist_ok=exist_ok, **flags)

    def picongpu_add_custom_user_input(self, custom_user_input: pypicongpu.customuserinput.CustomUserInput):
        """add custom user input to previously stored input"""
        self.picongpu_custom_user_input = (self.picongpu_custom_user_input or []) + [custom_user_input]

    def add_interaction(self, interaction) -> None:
        pypicongpu.util.unsupported(
            "PICMI standard interactions are not supported by PIConGPU, use the picongpu specific Interaction object instead"
        )

    # @todo add refactor once restarts are supported by the Runner, Brian Marre, 2024
    def step(self, nsteps: int = 1, **flags):
        if nsteps != self.max_steps:
            raise ValueError(
                "PIConGPU does not support stepwise running. Invoke step() with max_steps (={})".format(self.max_steps)
            )
        self.picongpu_run(**flags)

    def _generate_openpmd_plugins(self, diagnostics, num_steps, default_particle_shape=None):
        diagnostics = list(diagnostics)
        return [
            OpenPMDPlugin(
                sources=[
                    (
                        diagnostic.period.get_as_pypicongpu(time_step_size=self.time_step_size, num_steps=num_steps),
                        diagnostic.species.get_as_pypicongpu(default_particle_shape=default_particle_shape)
                        if isinstance(diagnostic, ParticleDump)
                        else PyPIConGPUFieldDump(
                            name=diagnostic.fieldname,
                            filtername=diagnostic.filtername,
                            species_name=None if isinstance(diagnostic, NativeFieldDump) else diagnostic.species_name,
                            functor=None
                            if isinstance(diagnostic, NativeFieldDump)
                            else diagnostic.functor.get_as_pypicongpu(mode="DerivedField"),
                        ),
                    )
                    for diagnostic in filter(lambda x: x.options == options, diagnostics)
                ],
                config=options,
            )
            for options in unique(map(lambda x: x.options, diagnostics))
        ]

    def _generate_plugins(self, num_steps, default_particle_shape=None):
        return [
            entry.get_as_pypicongpu(
                time_step_size=self.time_step_size,
                num_steps=num_steps,
                default_particle_shape=default_particle_shape,
            )
            for entry in self.diagnostics
            if not handled_via_openpmd(entry)
        ] + self._generate_openpmd_plugins(
            filter(handled_via_openpmd, self.diagnostics), num_steps, default_particle_shape
        )

    def _check_compatibility(self):
        pypicongpu.util.unsupported("verbose", self.verbose)
        pypicongpu.util.unsupported("gamma boost", self.gamma_boost)
        if len(self.laser_injection_methods) != self.laser_injection_methods.count(None):
            pypicongpu.util.unsupported("laser injection method", self.laser_injection_methods, [])
        if self.max_steps is None and self.max_time is None:
            raise ValueError("runtime not specified (neither as step count nor max time)")
        if isinstance(self.solver.grid, Cartesian2DGrid):
            # 2D3V: there is no spatial z coordinate (momentum still has all three components).
            for diagnostic in filter(lambda d: isinstance(d, PhaseSpace), self.diagnostics):
                if diagnostic.spatial_coordinate == "z":
                    raise ValueError(
                        "A phase-space diagnostic with spatial coordinate 'z' is not supported in 2D. "
                        f"You gave {diagnostic.spatial_coordinate=} on a 2D grid."
                    )
            for species in self.species:
                if isinstance(species.initial_distribution, AnalyticDistribution):
                    if Symbol("z") in species.initial_distribution.density_expression.free_symbols:
                        raise ValueError(
                            "A z-dependent AnalyticDistribution density is not supported on a 2D grid. "
                            f"You gave a density formula depending on 'z' for species {species.name!r} on a 2D grid."
                        )

    def _check_huygens_surface_positions(self):
        # Every laser renders into the single incidentField, so all lasers must
        # share one Huygens surface. Enforce strict list-equality of the three
        # [neg, pos] pairs, using the first laser as the reference
        # (https://github.com/chillenzer-agents/picongpu/issues/115).
        if len(self.lasers) <= 1:
            return
        reference = self.lasers[0].picongpu_huygens_surface_positions
        for laser in self.lasers[1:]:
            if laser.picongpu_huygens_surface_positions != reference:
                raise ValueError(
                    f"Inconsistent Huygens surface positions across lasers: {type(laser).__name__} has "
                    f"picongpu_huygens_surface_positions={laser.picongpu_huygens_surface_positions} but "
                    f"{type(self.lasers[0]).__name__} (the first laser) uses {reference}. "
                    "set every laser's `picongpu_huygens_surface_positions` to the same value."
                )

    def _collect_particle_filters(self):
        # This does not necessarily work on Binning plugin
        # because that might have a list of species.
        # But that's fine because the Binning plugin uses it's own mechanism
        # and we don't need their filters to register
        # unless they are used somewhere else as well.
        return unique(
            map(
                get_as_pypicongpu,
                chain(
                    UnpackChain(self).diagnostics.species.functor,
                    UnpackChain(self).picongpu_interaction.screening_species.functor,
                    UnpackChain(self).picongpu_interaction.collisions.species_pairs[:].functor,
                ),
            )
        )

    def get_as_pypicongpu(self) -> pypicongpu.simulation.Simulation:
        """translate to PyPIConGPU object"""
        self._check_compatibility()
        self._check_huygens_surface_positions()

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
        # We need to make sure that bare collisions are merged into a setup,
        # no matter if the interactions were assembled at construction time or later.
        self.picongpu_interaction = _validate_collisional_physics_setup(self.picongpu_interaction)
        # We provide the default as last element and we'll only read the first element:
        collisions = [x for x in self.picongpu_interaction if isinstance(x, CollisionalPhysicsSetup)] + [
            CollisionalPhysicsSetup()
        ]

        return pypicongpu.simulation.Simulation(
            species=map(
                lambda s: s.get_as_pypicongpu(default_particle_shape=self.particle_shape), sorted(self.species)
            ),
            init_operations=init_operations,
            typical_ppc=typical_ppc,
            delta_t_si=self.time_step_size,
            solver=self.solver.get_as_pypicongpu(),
            customuserinput=self.picongpu_custom_user_input,
            grid=self.solver.grid.get_as_pypicongpu(),
            binomial_current_interpolation=self.solver.source_smoother is not None,
            moving_window=moving_window,
            walltime=walltime or Walltime(walltime=datetime.timedelta(hours=1)),
            time_steps=time_steps,
            laser=[ll.get_as_pypicongpu() for ll in self.lasers] or None,
            output=self._generate_plugins(time_steps, self.particle_shape),
            particle_filters=self._collect_particle_filters(),
            base_density=self._get_base_density(),
            synchrotron_params=synchrotron_params[0],
            collisional_physics=collisions[0].get_as_pypicongpu(default_particle_shape=self.particle_shape),
            precision=self.picongpu_precision,
            precision_overrides=self.picongpu_precision_config.get_as_pypicongpu(),
            memory_config=self.picongpu_memory_config.get_as_pypicongpu(),
        )

    def _get_base_density(self) -> float:
        return self.picongpu_base_density or 1.0e25

    def run(self, *args, **kwargs) -> None:
        return self.picongpu_run(*args, **kwargs)

    def picongpu_run(self, setup_dir=None, run_dir=None, **flags) -> None:
        """build and run PIConGPU simulation"""
        runner = self.picongpu_get_runner(setup_dir=setup_dir, run_dir=run_dir)
        runner.generate(**flags)
        runner.run()

    def picongpu_get_runner(self, **kwargs) -> Runner:
        if self._runner is None:
            self._runner = Runner(
                **_drop_none(
                    dict(sim=self.get_as_pypicongpu(), template_dir=self.picongpu_template_dir or (templates.path(),))
                    | kwargs
                )
            )
        return self._runner

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


def _drop_none(d):
    return {key: value for key, value in d.items() if value is not None}
