"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

# make pypicongpu classes accessible for conversion to pypicongpu
from .. import pypicongpu
from .species import Species
from .interaction.ionization import IonizationModel

from . import constants
from .grid import Cartesian3DGrid
from .interaction import Interaction

import picmistandard

import math
import typeguard
import pathlib
import logging
import typing


# may not use pydantic since inherits from _DocumentedMetaClass
@typeguard.typechecked
class Simulation(picmistandard.PICMI_Simulation):
    """
    Simulation as defined by PICMI

    please refer to the PICMI documentation for the spec
    https://picmi-standard.github.io/standard/simulation.html
    """

    picongpu_custom_user_input = pypicongpu.util.build_typesafe_property(
        typing.Optional[list[pypicongpu.customuserinput.InterfaceCustomUserInput]]
    )
    """
    list of custom user input objects

    update using picongpu_add_custom_user_input() or by direct setting
    """

    picongpu_interaction = pypicongpu.util.build_typesafe_property(typing.Optional[Interaction])
    """Interaction instance containing all particle interactions of the simulation, set to None to have no interactions"""

    picongpu_typical_ppc = pypicongpu.util.build_typesafe_property(typing.Optional[int])
    """
    typical number of particle in a cell in the simulation

    used for normalization of code units

    optional, if set to None, will be set to median ppc of all species ppcs
    """

    picongpu_template_dir = pypicongpu.util.build_typesafe_property(typing.Optional[str])
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

    __runner = pypicongpu.util.build_typesafe_property(typing.Optional[pypicongpu.runner.Runner])

    # @todo remove boiler plate constructor argument list once picmistandard reference implementation switches to
    #   pydantic, Brian Marre, 2024
    def __init__(
        self,
        picongpu_template_dir: typing.Optional[typing.Union[str, pathlib.Path]] = None,
        picongpu_typical_ppc: typing.Optional[int] = None,
        picongpu_moving_window_move_point: typing.Optional[float] = None,
        picongpu_moving_window_stop_iteration: typing.Optional[int] = None,
        picongpu_interaction: typing.Optional[Interaction] = None,
        **keyword_arguments,
    ):
        if picongpu_template_dir is not None:
            self.picongpu_template_dir = str(picongpu_template_dir)
        else:
            self.picongpu_template_dir = picongpu_template_dir

        self.picongpu_typical_ppc = picongpu_typical_ppc
        self.picongpu_moving_window_move_point = picongpu_moving_window_move_point
        self.picongpu_moving_window_stop_iteration = picongpu_moving_window_stop_iteration
        self.picongpu_interaction = picongpu_interaction
        self.picongpu_custom_user_input = None
        self.__runner = None

        picmistandard.PICMI_Simulation.__init__(self, **keyword_arguments)

        # additional PICMI stuff checks, @todo move to picmistandard, Brian Marre, 2024
        ## throw if both cfl & delta_t are set
        if self.solver is not None and "Yee" == self.solver.method and isinstance(self.solver.grid, Cartesian3DGrid):
            self.__yee_compute_cfl_or_delta_t()

        # checks on picongpu specific stuff
        ## template_path is valid
        if picongpu_template_dir == "":
            raise ValueError("picongpu_template_dir MUST NOT be empty string")
        if picongpu_template_dir is not None:
            template_path = pathlib.Path(picongpu_template_dir)
            if not template_path.is_dir():
                raise ValueError("picongpu_template_dir must be existing directory")

    def __yee_compute_cfl_or_delta_t(self) -> None:
        """
        use delta_t or cfl to compute the other

        needs grid parameters for computation
        Only works if method is Yee.

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
        assert "Yee" == self.solver.method
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

    def __get_operations_simple_density(
        self,
        pypicongpu_by_picmi_species: typing.Dict[Species, pypicongpu.species.Species],
    ) -> typing.List[pypicongpu.species.operation.SimpleDensity]:
        """
        retrieve operations for simple density placements

        Initialized Position & Weighting based on picmi initial distribution &
        layout.

        initializes species using the same layout & profile from the same
        operation
        """
        # species with the same layout and initial distribution will result in
        # the same macro particle placement
        # -> throw them into a single operation
        picmi_species_by_profile_by_layout = {}
        for picmi_species, layout in zip(self.species, self.layouts):
            if layout is None or picmi_species.initial_distribution is None:
                # not placed -> not handled here
                continue

            if layout not in picmi_species_by_profile_by_layout:
                picmi_species_by_profile_by_layout[layout] = {}

            profile = picmi_species.initial_distribution
            if profile not in picmi_species_by_profile_by_layout[layout]:
                picmi_species_by_profile_by_layout[layout][profile] = []

            picmi_species_by_profile_by_layout[layout][profile].append(picmi_species)

        # re-group as operations
        all_operations = []
        for (
            layout,
            picmi_species_by_profile,
        ) in picmi_species_by_profile_by_layout.items():
            for profile, picmi_species_list in picmi_species_by_profile.items():
                assert isinstance(layout, picmistandard.PICMI_PseudoRandomLayout)

                op = pypicongpu.species.operation.SimpleDensity()
                op.ppc = layout.n_macroparticles_per_cell
                op.profile = profile.get_as_pypicongpu()

                op.species = set(
                    map(
                        lambda picmi_species: pypicongpu_by_picmi_species[picmi_species],
                        picmi_species_list,
                    )
                )

                all_operations.append(op)

        return all_operations

    def __get_operations_not_placed(
        self,
        pypicongpu_by_picmi_species: typing.Dict[Species, pypicongpu.species.Species],
    ) -> typing.List[pypicongpu.species.operation.NotPlaced]:
        """
        retrieve operations for not placed species

        Problem: PIConGPU species need a position. But to get a position
        generated, a species needs an operation which provides this position.
        (E.g. SimpleDensity for regular profiles.)

        Solution: If a species has no initial distribution (profile), the
        position attribute is provided by a NotPlaced operator, which does not
        create any macroparticles (during initialization, that is). However,
        using other methods (electron spawning...) macrosparticles can be
        created by PIConGPU itself.
        """
        all_operations = []

        for picmi_species, layout in zip(self.species, self.layouts):
            if layout is not None or picmi_species.initial_distribution is not None:
                continue

            # is not placed -> add op
            not_placed = pypicongpu.species.operation.NotPlaced()
            not_placed.species = pypicongpu_by_picmi_species[picmi_species]
            all_operations.append(not_placed)

        return all_operations

    def __get_operations_from_individual_species(
        self,
        pypicongpu_by_picmi_species: typing.Dict[Species, pypicongpu.species.Species],
    ) -> typing.List[pypicongpu.species.operation.Operation]:
        """
        call get_independent_operations() of all species

        used for momentum: Momentum depends only on temperature & drift, NOT on
        other species. Therefore, the generation of the momentum operations is
        performed inside of the individual species objects.
        """
        all_operations = []

        for picmi_species, pypicongpu_species in pypicongpu_by_picmi_species.items():
            all_operations += picmi_species.get_independent_operations(pypicongpu_species, self.picongpu_interaction)

        return all_operations

    def __check_preconditions_init_manager(self) -> None:
        """check preconditions, @todo move to picmistandard, Brian Marre 2024"""
        assert len(self.species) == len(self.layouts)

        for layout, picmi_species in zip(self.layouts, self.species):
            profile = picmi_species.initial_distribution
            ratio = picmi_species.density_scale

            assert 1 != [layout, profile].count(
                None
            ), "species need BOTH layout AND initial distribution set (or neither)"

            if ratio is not None:
                assert (
                    layout is not None and profile is not None
                ), "layout and initial distribution must be set to use density scale"

    def __get_translated_species_and_ionization_models(
        self,
    ) -> tuple[
        dict[Species, pypicongpu.species.Species],
        dict[Species, None | dict[IonizationModel, pypicongpu.species.constant.ionizationmodel.IonizationModel]],
    ]:
        """
        get mappping of PICMI species to PyPIConGPU species and mapping of of simulation

        @details cache to reuse *exactly the same* object in operations
        """

        pypicongpu_by_picmi_species = {}
        ionization_model_conversion_by_species = {}
        for picmi_species in self.species:
            # @todo split into two different fucntion calls?, Brian Marre, 2024
            pypicongpu_species, ionization_model_conversion = picmi_species.get_as_pypicongpu(self.picongpu_interaction)

            pypicongpu_by_picmi_species[picmi_species] = pypicongpu_species
            ionization_model_conversion_by_species[picmi_species] = ionization_model_conversion

        return pypicongpu_by_picmi_species, ionization_model_conversion_by_species

    def __fill_in_ionization_electrons(
        self,
        pypicongpu_by_picmi_species: dict[Species, pypicongpu.species.Species],
        ionization_model_conversion_by_species: dict[
            Species, None | dict[IonizationModel, pypicongpu.species.constant.ionizationmodel.IonizationModel]
        ],
    ) -> None:
        """
        set the ionization electron species for each ionization model

        Ionization electron species need to be set after species translation is complete since the PyPIConGPU electron
        species is not at the time of translation by the PICMI ion species.
        """
        if self.picongpu_interaction is not None:
            self.picongpu_interaction.fill_in_ionization_electron_species(
                pypicongpu_by_picmi_species, ionization_model_conversion_by_species
            )

    def __get_init_manager(self) -> pypicongpu.species.InitManager:
        """
        create & fill an Initmanager

        performs the following steps:
        1. check preconditions
        2. translate species and ionization models to PyPIConGPU representations
           Note: Cache translations to avoid creating new translations by continuously translating again and again
        3. generate operations which have inter-species dependencies
        4. generate operations without inter-species dependencies
        """
        self.__check_preconditions_init_manager()
        (
            pypicongpu_by_picmi_species,
            ionization_model_conversion_by_species,
        ) = self.__get_translated_species_and_ionization_models()

        # fill inter-species dependencies
        self.__fill_in_ionization_electrons(pypicongpu_by_picmi_species, ionization_model_conversion_by_species)

        # init PyPIConGPU init manager
        initmgr = pypicongpu.species.InitManager()

        for pypicongpu_species in pypicongpu_by_picmi_species.values():
            initmgr.all_species.append(pypicongpu_species)

        # operations on multiple species
        initmgr.all_operations += self.__get_operations_simple_density(pypicongpu_by_picmi_species)

        # operations on single species
        initmgr.all_operations += self.__get_operations_not_placed(pypicongpu_by_picmi_species)
        initmgr.all_operations += self.__get_operations_from_individual_species(pypicongpu_by_picmi_species)

        return initmgr

    def write_input_file(
        self, file_name: str, pypicongpu_simulation: typing.Optional[pypicongpu.simulation.Simulation] = None
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

    def picongpu_add_custom_user_input(self, custom_user_input: pypicongpu.customuserinput.InterfaceCustomUserInput):
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

    def get_as_pypicongpu(self) -> pypicongpu.simulation.Simulation:
        """translate to PyPIConGPU object"""
        s = pypicongpu.simulation.Simulation()

        s.delta_t_si = self.time_step_size
        s.solver = self.solver.get_as_pypicongpu()

        # already pypicongpu objects, therefore directly passing on
        s.custom_user_input = self.picongpu_custom_user_input

        # calculate time step
        if self.max_steps is not None:
            s.time_steps = self.max_steps
        elif self.max_time is not None:
            s.time_steps = self.max_time / self.time_step_size
        else:
            raise ValueError("runtime not specified (neither as step count nor max time)")

        pypicongpu.util.unsupported("verbose", self.verbose)
        pypicongpu.util.unsupported("particle shape", self.particle_shape, "linear")
        pypicongpu.util.unsupported("gamma boost, use picongpu_moving_window_move_point instead", self.gamma_boost)

        try:
            s.grid = self.solver.grid.get_as_pypicongpu()
        except AttributeError:
            pypicongpu.util.unsupported(f"grid type: {type(self.solver.grid)}")

        # any injection method != None is not supported
        if len(self.laser_injection_methods) != self.laser_injection_methods.count(None):
            pypicongpu.util.unsupported("laser injection method", self.laser_injection_methods, [])

        # pypicongpu interface currently only supports one laser, @todo change Brian Marre, 2024
        if len(self.lasers) > 1:
            pypicongpu.util.unsupported("more than one laser")

        if len(self.lasers) == 1:
            # check requires grid, so grid is translated (and thereby also checked) above
            s.laser = self.lasers[0].get_as_pypicongpu()
        else:
            # explictly disable laser (as required by pypicongpu)
            s.laser = None

        s.init_manager = self.__get_init_manager()

        # set typical ppc if not set explicitly by user
        if self.picongpu_typical_ppc is None:
            s.typical_ppc = (s.init_manager).get_typical_particle_per_cell()
        else:
            s.typical_ppc = self.picongpu_typical_ppc

        if s.typical_ppc < 1:
            raise ValueError("typical_ppc must be >= 1")

        # disable moving Window if explicitly activated by the user
        if self.picongpu_moving_window_move_point is None:
            s.moving_window = None
        else:
            s.moving_window = pypicongpu.movingwindow.MovingWindow(
                move_point=self.picongpu_moving_window_move_point,
                stop_iteration=self.picongpu_moving_window_stop_iteration,
            )

        return s

    def picongpu_run(self) -> None:
        """build and run PIConGPU simulation"""
        if self.__runner is None:
            self.__runner = pypicongpu.runner.Runner(self.get_as_pypicongpu(), self.picongpu_template_dir)
        self.__runner.generate()
        self.__runner.build()
        self.__runner.run()

    def picongpu_get_runner(self) -> pypicongpu.runner.Runner:
        if self.__runner is None:
            self.__runner = pypicongpu.runner.Runner(self.get_as_pypicongpu(), self.picongpu_template_dir)
        return self.__runner
