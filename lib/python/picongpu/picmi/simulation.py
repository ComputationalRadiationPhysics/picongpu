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
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from picongpu import pypicongpu, templates
from picongpu.picmi import constants
from picongpu.picmi.applied_field import AnyAppliedField, combine_applied_fields
from picongpu.picmi.diagnostics.field_dump import NativeFieldDump, _FieldDump
from picongpu.picmi.diagnostics.optical_imaging import OpticalImaging
from picongpu.picmi.diagnostics.particle_dump import ParticleDump
from picongpu.picmi.diagnostics.phase_space import PhaseSpace
from picongpu.picmi.distribution.AnalyticDistribution import AnalyticDistribution
from picongpu.picmi.distribution.FromFileDistribution import FromFileDistribution
from picongpu.picmi.grid import Cartesian2DGrid, Cartesian3DGrid
from picongpu.picmi.interaction import (
    SUPPORTED_INTERACTION_TYPES,
    Interaction,
    PICMI_Interaction,
    Synchrotron,
)
from picongpu.picmi.interaction.collision import Collision, CollisionalPhysicsSetup
from picongpu.picmi.interaction.ionization.fieldionization import FieldIonization
from picongpu.picmi.species import _STANDARD_SHAPES
from picongpu.picmi.species_requirements import (
    ParticleFromFileOperation,
    SimpleDensityOperation,
    SimpleMomentumOperation,
    get_as_pypicongpu,
    resolving_add,
    run_construction,
)
from picongpu.picmi.memory_config import MemoryConfig
from picongpu.picmi.precision_config import PrecisionConfig
from picongpu.pypicongpu.backgroundfield import BackgroundField as PyPIConGPUBackgroundField
from picongpu.pypicongpu.output.openpmd_plugin import FieldDump as PyPIConGPUFieldDump
from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDPlugin
from picongpu.pypicongpu.runner import Runner
from picongpu.pypicongpu.species.attribute.momentum import Momentum
from picongpu.pypicongpu.particle_functor.particle_functor import _SpeciesName
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams
from picongpu.pypicongpu.util import UnpackChain, unique
from picongpu.pypicongpu.walltime import Walltime


def _is_multi_species(entry):
    """Whether a ``species`` entry is a PICMI ``MultiSpecies`` group."""
    return isinstance(entry, picmistandard.PICMI_MultiSpecies)


def _entry_members(entry):
    """The plain species making up a ``species`` entry, in order."""
    return entry.species_instances_list if _is_multi_species(entry) else [entry]


def _register_density_operation(entry, layout, grid):
    """Register the density (and momentum) requirements of one species entry.

    A whole :class:`picmi.MultiSpecies` entry is treated as one coordinated group:
    all of its members share a single :class:`SimpleDensityOperation`, so they
    are placed on identical in-cell positions (collective/charge-neutral
    initialisation). A standalone species is always its own operation
    (independent by default). Each member additionally gets its per-species
    momentum initialisation.
    """
    members = _entry_members(entry)
    profile = members[0].initial_distribution
    if profile is None:
        return
    if isinstance(profile, FromFileDistribution):
        # Each from-file species has its own file and iteration; there is no
        # coordinated density placement across a group.
        for member in members:
            member.register_requirements(
                [
                    Momentum(),
                    Weighting(),
                    ParticleFromFileOperation(
                        species=member,
                        file_path=member.initial_distribution.file_path,
                        iteration=member.initial_distribution.iteration,
                    ),
                ]
            )
        return
    members[0].register_requirements([Weighting(), SimpleDensityOperation(species=members, layout=layout, grid=grid)])
    for member in members:
        member.register_requirements([Momentum(), SimpleMomentumOperation(species=member)])


def _validate_species_layout(entry, layout):
    """Validate one ``(species entry, layout)`` pair.

    Shared by the imperative ``add_species`` path and the declarative
    constructor path. Validation is per member: a ``MultiSpecies`` group passes
    one layout for all members.
    """
    members = _entry_members(entry)
    if len(members) > 1 and any(isinstance(m.initial_distribution, FromFileDistribution) for m in members):
        # A MultiSpecies shares one distribution across its members; a from-file
        # distribution cannot satisfy two differently-named species from a single
        # file (each member would look up its own species name in the same file).
        raise ValueError(
            "A from-file distribution cannot be combined with a MultiSpecies. "
            "A single external file cannot supply several differently-named species; "
            "add each from-file species as its own species entry instead."
        )
    for species in members:
        if isinstance(species.initial_distribution, FromFileDistribution):
            # The particle positions come from the file; a layout would be
            # silently ignored, so reject it explicitly.
            if layout is not None:
                raise ValueError(
                    "A from-file distribution determines the particle positions itself and cannot be combined "
                    f"with a layout. You gave {species.initial_distribution=} but {layout=}."
                )
            # The weightings come from the file, so density_scale would be
            # silently ignored; reject it instead.
            if species.density_scale is not None:
                raise ValueError(
                    "density_scale cannot be combined with a from-file distribution: the particle weightings are "
                    f"read from the file. You gave {species.density_scale=}."
                )
            continue
        if species.density_scale is not None and (layout is None and species.initial_distribution is None):
            raise ValueError("layout and initial distribution must be set to use density scale")
        if layout is not None and species.initial_distribution is None:
            raise ValueError(
                f"An initial distribution needs a layout. You've given {layout=} but {species.initial_distribution=}."
            )
        if species.initial_distribution is not None and layout is None:
            raise ValueError(
                f"A species with an initial distribution needs a layout. "
                f"You gave {species.initial_distribution=} but {layout=}."
            )


def _validate_lengths(species, layouts):
    if len(layouts) != len(species):
        raise ValueError(
            f"species and layouts must have the same length, but you gave {len(species)=} and {len(layouts)=}."
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


def _translate_laser(laser, pypicongpu_grid):
    """Translate one PICMI laser, handing it the grid for the reference-frame conversion.

    All closed-form PICMI lasers need the grid (cell size + domain cell counts)
    to convert their centroid-based definition into the core's Huygens-surface
    convention when computing ``pulse_init``. Laser types that do not (e.g.
    FromOpenPMDPulse, which carries its own time offset) simply ignore the
    extra arguments.
    """
    return laser.get_as_pypicongpu(pypicongpu_grid.cell_size, pypicongpu_grid.cell_cnt)


def _normalise_interaction(interaction):
    """Map a standard interaction onto the equivalent PIConGPU interaction.

    ``picmi.FieldIonization`` (and the plain standard
    ``picmistandard.PICMI_FieldIonization``) are converted to the matching
    concrete PIConGPU field ionization model. Everything else is returned
    unchanged and validated below.
    """
    if isinstance(interaction, FieldIonization):
        return interaction.get_as_pypicongpu()
    if isinstance(interaction, picmistandard.PICMI_FieldIonization):
        # A plain standard object carries only model/ionized_species/product_species;
        # the conversion reports the required-knob errors for ADK/BSI.
        return FieldIonization(
            model=interaction.model,
            ionized_species=interaction.ionized_species,
            product_species=interaction.product_species,
        ).get_as_pypicongpu()
    if isinstance(interaction, PICMI_Interaction) and not isinstance(interaction, SUPPORTED_INTERACTION_TYPES):
        pypicongpu.util.unsupported("This PICMI interaction type is not supported by PIConGPU", interaction)
    return interaction


def _prepare_interactions(interactions):
    """Normalise and validate the interaction list.

    This is the single pipeline shared by the constructor ``interactions=[...]``
    parameter and :meth:`add_interaction`: standard field ionization is mapped
    to PIConGPU's concrete model, and bare collisions are merged into a
    ``CollisionalPhysicsSetup``.
    """
    return _validate_collisional_physics_setup([_normalise_interaction(x) for x in interactions])


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

    interactions: Annotated[list[Interaction | PICMI_Interaction], BeforeValidator(_prepare_interactions)] = Field(
        default_factory=list
    )
    """
    All particle interactions of the simulation.

    The standard ``interactions=[...]`` constructor parameter and
    :meth:`add_interaction` are the entry points. They accept every interaction
    type PIConGPU supports: field ionization (``picmi.FieldIonization``),
    collisions (``picmi.Collision``/``picmi.CollisionalPhysicsSetup``),
    synchrotron radiation (``picmi.Synchrotron``) and the concrete ionization
    models.
    """

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

    def _validate_min_weighting(value: float | None) -> float | None:
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"Minimum weighting must be finite and > 0, not {value=}.")
        return value

    picongpu_min_weighting: Annotated[float | None, AfterValidator(_validate_min_weighting)] = Field(default=None)
    """
    minimum macro-particle weighting below which particles are not created / are deleted

    unit: none (bare float in PIConGPU code units, not SI)

    optional; if set to None, PIConGPU's default of 10.0 is used
    """

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

    def model_post_init(self, __context) -> None:
        # Honour the documented declarative constructor style
        # ``Simulation(species=[...], layouts=[...])`` once, at construction.
        # The entries are stored AS GIVEN (a ``MultiSpecies`` stays one entry,
        # it is not expanded into its members); translation maps each entry onto
        # one density operation. Unlike an after-validator this hook does not run
        # on later assignments, so the inherited ``add_species_through_plane``
        # (whose base ``_append`` sets ``species`` and ``layouts`` in two
        # separate steps) is unaffected.
        if self.species or self.layouts:
            _validate_lengths(self.species, self.layouts)
            for entry, layout in zip(self.species, self.layouts):
                _validate_species_layout(entry, layout)
                _register_density_operation(entry, layout, self.solver.grid)

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
        """
        Add an interaction to the simulation.

        Accepts every interaction type PIConGPU supports: field ionization
        (``picmi.FieldIonization`` or the plain standard
        ``picmistandard.PICMI_FieldIonization``, both converted to the matching
        concrete model) as well as PIConGPU's own collision,
        collisional-physics-setup and synchrotron objects.

        Equivalent to appending to the ``interactions=[...]`` constructor
        parameter; both go through the same pipeline.
        """
        self.interactions = [*(self.interactions or []), interaction]

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
            # A Huygens surface cannot be placed on a Z face either (answer 5 of
            # https://github.com/chillenzer-agents/picongpu/issues/180). Only
            # standard lasers carry an entry-face selection (TWTS and
            # fromOpenPMDPulse keep their dedicated placement).
            for laser in self.lasers:
                if (validate := getattr(laser, "validate_entry_faces", None)) is not None:
                    validate(2)
            optical_imaging = [d for d in self.diagnostics if isinstance(d, OpticalImaging)]
            if optical_imaging:
                raise ValueError(
                    "An OpticalImaging/Shadowgraphy diagnostic requires a 3D simulation (it extracts a slice in "
                    "the z direction), but you configured a 2D grid. "
                    f"You gave {len(optical_imaging)} such diagnostic(s) on a 2D grid."
                )
            for diagnostic in filter(lambda d: isinstance(d, PhaseSpace), self.diagnostics):
                if diagnostic.spatial_coordinate == "z":
                    raise ValueError(
                        "A phase-space diagnostic with spatial coordinate 'z' is not supported in 2D. "
                        f"You gave {diagnostic.spatial_coordinate=} on a 2D grid."
                    )
            for entry in self.species:
                for species in _entry_members(entry):
                    if isinstance(species.initial_distribution, AnalyticDistribution):
                        if species.initial_distribution.dim == 3:
                            raise ValueError(
                                "A z-dependent AnalyticDistribution density is not supported on a 2D grid. "
                                f"You gave a density formula depending on 'z' for species {species.name!r} on a 2D grid."
                            )
                    if isinstance(species.initial_distribution, FromFileDistribution):
                        # A dimension mismatch (e.g. a 3D file into a 2D simDim)
                        # is rejected until a principled mapping is decided.
                        raise ValueError(
                            "FromFileDistribution is not supported on a 2D grid yet. "
                            f"You gave it for species {species.name!r} on a 2D grid."
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
        # Collect every reusable filter with the compile-time name(s) of the species
        # it is registered on (merged across its occurrences). particleFilters.param
        # then emits a name-keyed SpeciesEligibleForSolver specialisation narrowing
        # the filter to exactly those species instead of all of VectorAllSpecies.
        # Functors are deduplicated by value, like before: they are not hashable
        # because of their volatile per-call ``typename``.
        registered = list(
            zip(
                map(
                    get_as_pypicongpu,
                    chain(
                        UnpackChain(self).diagnostics.species.functor,
                        UnpackChain(self).interactions.screening_species.functor,
                        UnpackChain(self).interactions.collisions.species_pairs[:].functor,
                    ),
                ),
                chain(
                    UnpackChain(self).diagnostics.species.species_name,
                    UnpackChain(self).interactions.screening_species.species_name,
                    UnpackChain(self).interactions.collisions.species_pairs[:].species_name,
                ),
            )
        )
        return [
            functor.model_copy(
                update={
                    "species_names": [
                        _SpeciesName(name=name) for name in sorted({n for f, n in registered if f == functor})
                    ]
                }
            )
            for functor in unique(f for f, _ in registered)
        ]

    def _register_functor_requirements(self):
        """Register the species attributes each diagnostic functor accesses.

        Diagnostics own both the functor and the species it runs on, so this is
        the point where ``Species.register_requirements`` can be handed the
        functor. The functor translates its accessed attributes (e.g.
        ``momentumPrev1``) into the corresponding ``Attribute`` requirements.
        Particle filters are registered from ``FilteredSpecies`` instead.
        """
        for diagnostic in self.diagnostics:
            functors = [
                functor
                for name in ("functor", "deposition_functor")
                if (functor := getattr(diagnostic, name, None)) is not None
            ] + [axis.functor for axis in getattr(diagnostic, "axes", None) or []]
            if not functors:
                continue
            species = getattr(diagnostic, "species", None)
            for one_species in species if isinstance(species, list) else [species]:
                if one_species is None:
                    continue
                while hasattr(one_species, "species"):
                    # A FilteredSpecies wraps the owner species; its filter
                    # accesses attributes just like the diagnostic functor, so
                    # register them on the (eventual) owner as well. This is
                    # needed for diagnostics (e.g. DerivedFieldDump) that only
                    # read the species by name and never convert the wrapper.
                    if (filter_functor := getattr(one_species, "functor", None)) is not None:
                        one_species.species.register_requirements(filter_functor.get_required_attributes())
                    one_species = one_species.species
                for functor in functors:
                    one_species.register_requirements(functor.get_required_attributes())

    def get_as_pypicongpu(self) -> pypicongpu.simulation.Simulation:
        """translate to PyPIConGPU object"""
        self._check_compatibility()
        self._check_huygens_surface_positions()
        self._register_functor_requirements()

        all_species = chain.from_iterable(_entry_members(entry) for entry in self.species)
        init_operations = organise_init_operations(
            chain(*(s.get_operation_requirements() for s in sorted(all_species)))
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
            [x.synchrotron_parameters for x in self.interactions if isinstance(x, Synchrotron)]
        ) + [SynchrotronParams()]
        if len(synchrotron_params) > 2:
            raise ValueError(
                f"You have configured the Synchrotron extension multiple times with different arguments. This is not allowed! You gave {synchrotron_params[:-1]=}."
            )
        # We need to make sure that bare collisions are merged into a setup,
        # no matter if the interactions were assembled at construction time or later.
        self.interactions = _validate_collisional_physics_setup(self.interactions)
        # We provide the default as last element and we'll only read the first element:
        collisions = [x for x in self.interactions if isinstance(x, CollisionalPhysicsSetup)] + [
            CollisionalPhysicsSetup()
        ]

        pypicongpu_grid = self.solver.grid.get_as_pypicongpu()
        # The laser frontend only exposes the centroid-based PICMI reference
        # frame; converting to the core's Huygens-surface convention needs the
        # grid, which is only known here at translation time.
        return pypicongpu.simulation.Simulation(
            species=map(
                lambda s: s.get_as_pypicongpu(
                    default_particle_shape=self.particle_shape,
                    time_step_size=self.time_step_size,
                    num_steps=time_steps,
                ),
                sorted(chain.from_iterable(_entry_members(entry) for entry in self.species)),
            ),
            init_operations=init_operations,
            typical_ppc=typical_ppc,
            delta_t_si=self.time_step_size,
            solver=self.solver.get_as_pypicongpu(),
            customuserinput=self.picongpu_custom_user_input,
            grid=pypicongpu_grid,
            binomial_current_interpolation=self.solver.source_smoother is not None,
            moving_window=moving_window,
            walltime=walltime or Walltime(walltime=datetime.timedelta(hours=1)),
            time_steps=time_steps,
            laser=[_translate_laser(ll, pypicongpu_grid) for ll in self.lasers] or None,
            background_field=self._get_background_field(),
            output=self._generate_plugins(time_steps, self.particle_shape),
            particle_filters=self._collect_particle_filters(),
            base_density=self._get_base_density(),
            synchrotron_params=synchrotron_params[0],
            collisional_physics=collisions[0].get_as_pypicongpu(default_particle_shape=self.particle_shape),
            min_weighting=self.picongpu_min_weighting,
            precision=self.picongpu_precision,
            precision_overrides=self.picongpu_precision_config.get_as_pypicongpu(),
            memory_config=self.picongpu_memory_config.get_as_pypicongpu(),
        )

    def _get_base_density(self) -> float:
        return self.picongpu_base_density or 1.0e25

    def _get_background_field(self) -> PyPIConGPUBackgroundField | None:
        """
        Translate the configured applied fields into a single pypicongpu background field.

        The C++ core only evaluates one ``FieldBackgroundE``/``FieldBackgroundB``
        functor pair, so the contributions of all applied fields are summed per
        component. The influence knobs of the individual fields must agree, as
        they configure that single pair.
        """
        unsupported = [f for f in self.applied_fields if not isinstance(f, AnyAppliedField)]
        if unsupported:
            pypicongpu.util.unsupported(
                "applied fields other than ConstantAppliedField and AnalyticAppliedField", unsupported
            )
        if not self.applied_fields:
            return None
        return combine_applied_fields(self.applied_fields)

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

    def _picongpu_add_species(self, species, layout, initialize_self_field=None):
        # PICMI-standard interface: a ``MultiSpecies`` is added as one object
        # together with one layout for the whole group. The entry is stored AS
        # GIVEN (not expanded into its members); translation maps it onto one
        # density operation covering all its members.
        self.species.append(species)
        self.layouts.append(layout)
        _validate_species_layout(species, layout)
        _register_density_operation(species, layout, self.solver.grid)

    def add_species(self, species, layout, initialize_self_field=None):
        return self._picongpu_add_species(species, layout, initialize_self_field)


def organise_init_operations(operations):
    # materialise -- resolving_add is multi-pass
    operations = list(operations)
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
