"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ..pypicongpu import simulation, runner, util, species
from . import constants
from .grid import Cartesian3DGrid
from .species import Species as PicongpuPicmiSpecies

import picmistandard

from math import sqrt, isclose
from typeguard import typechecked
import pathlib
import logging
import typing


@typechecked
class Simulation(picmistandard.PICMI_Simulation):
    """
    Simulation as defined by PICMI

    please refer to the PICMI documentation for the spec
    https://picmi-standard.github.io/standard/simulation.html
    """

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

        delta_x = ((self.solver.grid.upper_bound[0]
                    - self.solver.grid.lower_bound[0])
                   / self.solver.grid.number_of_cells[0])
        delta_y = ((self.solver.grid.upper_bound[1]
                    - self.solver.grid.lower_bound[1])
                   / self.solver.grid.number_of_cells[1])
        delta_z = ((self.solver.grid.upper_bound[2]
                    - self.solver.grid.lower_bound[2])
                   / self.solver.grid.number_of_cells[2])

        if self.time_step_size is not None and \
                self.solver.cfl is not None:
            # both cfl & delta_t given -> check their compatibility
            delta_t_from_cfl = \
                self.solver.cfl / (constants.c
                                   * sqrt(1/delta_x**2
                                          + 1/delta_y**2
                                          + 1/delta_z**2))

            if delta_t_from_cfl != self.time_step_size:
                raise ValueError(
                    "time step size (delta t) does not match CFL "
                    "(Courant-Friedrichs-Lewy) parameter! delta_t: {}; "
                    "expected from CFL: {}"
                    .format(self.time_step_size, delta_t_from_cfl))
        else:
            if self.time_step_size is not None:
                # calculate cfl
                self.solver.cfl = \
                    self.time_step_size * (constants.c
                                           * sqrt(1/delta_x**2
                                                  + 1/delta_y**2
                                                  + 1/delta_z**2))
            elif self.solver.cfl is not None:
                # calculate delta_t
                self.time_step_size = \
                    self.solver.cfl / (constants.c
                                       * sqrt(1/delta_x**2
                                              + 1/delta_y**2
                                              + 1/delta_z**2))

            # if neither delta_t nor cfl are given simply silently pass
            # (might change in the future)

    def __init__(self,
                 picongpu_template_dir: typing.Optional[
                     typing.Union[str, pathlib.Path]] = None,
                 picongpu_typical_ppc: typing.Optional[int] = None,
                 **kw):
        # delegate actual work to parent
        super().__init__(**kw)

        # perform some additional checks on inputs

        # note: may throw if both cfl & delta_t are set
        if self.solver is not None and \
           "Yee" == self.solver.method and \
           isinstance(self.solver.grid, Cartesian3DGrid):
            self.__yee_compute_cfl_or_delta_t()

        if picongpu_template_dir is None:
            self.picongpu_template_dir = None
        else:
            assert "" != picongpu_template_dir, \
                "picongpu_template_dir MUST NOT be empty"
            # note: pathlib.Path(pathlib.Path(...)) is valid
            template_path = pathlib.Path(picongpu_template_dir)
            assert template_path.is_dir(), \
                "picongpu_template_dir must be existing dir"
            self.picongpu_template_dir = str(template_path)

        self.picongpu_typical_ppc = picongpu_typical_ppc

        # store runner state
        self.__runner = None

        self.__electron_species = None

    def __get_operations_simple_density(
            self,
            pypicongpu_by_picmi_species:
            typing.Dict[picmistandard.PICMI_Species, species.Species]) -> \
            typing.List[species.operation.SimpleDensity]:
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

            picmi_species_by_profile_by_layout[layout][profile].append(
                picmi_species)

        # re-group as operations
        all_operations = []
        for layout, picmi_species_by_profile in \
                picmi_species_by_profile_by_layout.items():
            for profile, picmi_species_list in \
                    picmi_species_by_profile.items():
                assert isinstance(layout,
                                  picmistandard.PICMI_PseudoRandomLayout)

                op = species.operation.SimpleDensity()
                op.ppc = layout.n_macroparticles_per_cell
                op.profile = profile.get_as_pypicongpu()

                op.species = set(map(
                    lambda picmi_species: pypicongpu_by_picmi_species[
                        picmi_species],
                    picmi_species_list))

                all_operations.append(op)

        return all_operations

    def __get_operations_not_placed(
            self,
            pypicongpu_by_picmi_species:
            typing.Dict[picmistandard.PICMI_Species, species.Species]) -> \
            typing.List[species.operation.NotPlaced]:
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
            if layout is not None or \
               picmi_species.initial_distribution is not None:
                continue

            # is not placed -> add op
            not_placed = species.operation.NotPlaced()
            not_placed.species = pypicongpu_by_picmi_species[picmi_species]
            all_operations.append(not_placed)

        return all_operations

    def __get_operations_from_individual_species(
            self,
            pypicongpu_by_picmi_species:
            typing.Dict[picmistandard.PICMI_Species, species.Species]) -> \
            typing.List[species.operation.Operation]:
        """
        call get_independent_operations() of all species

        used for momentum: Momentum depends only on temperature & drift, NOT on
        other species. Therefore, the generation of the momentum operations is
        performed inside of the individual species objects.
        """
        all_operations = []

        for picmi_species, pypicongpu_species in \
                pypicongpu_by_picmi_species.items():
            all_operations += \
                picmi_species.get_independent_operations(pypicongpu_species)

        return all_operations

    def __fill_ionization_electrons(
            self,
            pypicongpu_by_picmi_species:
            typing.Dict[picmistandard.PICMI_Species, species.Species]) -> None:
        """
        copy used-electron-relationship from PICMI to PIConGPU species

        Translating a PICMI species to a PyPIConGPU species creates a ionizers
        constant, but the reference to the used species is missing at this
        point (b/c the translated species doesn't know the corresponding
        PyPIConGPU species to be associated to.)

        This method fills the pypicongpu ionizers electron_species from the
        PICMI picongpu_ionization_electrons attribute.
        Note that for this the picongpu_ionization_electrons attribute must be
        already set, probably from __resolve_electrons()

        (An b/c python uses pointers, this will be applied to the existing
        species objects passed in pypicongpu_by_picmi_species)
        """

        for picmi_species, pypic_species in \
                pypicongpu_by_picmi_species.items():
            # only fill ionization electrons if required (by ionizers)
            if not pypic_species.has_constant_of_type(
                    species.constant.Ionizers):
                continue

            assert picmi_species.picongpu_ionization_electrons in \
                pypicongpu_by_picmi_species, "species {} (set as electrons " \
                "for species {} via picongpu_ionization_species) must be " \
                "explicitly added with add_species()".format(
                    picmi_species.picongpu_ionization_electrons.name,
                    pypic_species.name)

            ionizers = pypic_species.get_constant_by_type(
                species.constant.Ionizers)
            # is pointer -> sets correct species for actual pypicongpu species
            ionizers.electron_species = pypicongpu_by_picmi_species[
                picmi_species.picongpu_ionization_electrons]

    def __get_init_manager(self) -> species.InitManager:
        """
        create & fill an initmanager

        performs the following steps:
        1. check preconditions
        2. translate species to pypicongpu representation
           Note: Cache translations to avoid creating new translations by
           continuosly translating again and again
        3. generate operations which have inter-species dependencies
        4. generate operations without inter-species dependencies
        """
        initmgr = species.InitManager()

        # check preconditions
        assert len(self.species) == len(self.layouts)

        # either no layout AND no profile, or both
        # (also: no ratio without layout and profile)
        for layout, picmi_species in zip(self.layouts, self.species):
            profile = picmi_species.initial_distribution
            ratio = picmi_species.density_scale

            # either both None or both not None:
            assert 1 != [layout, profile].count(None), "species need BOTH " \
                "layout AND initial distribution set (or neither)"

            # ratio only set if
            if ratio is not None:
                assert layout is not None and profile is not None, "layout " \
                    "and initial distribution must be set to use density scale"

        # get species list
        ##

        # note: cache to reuse *exactly the same* object in operations
        pypicongpu_by_picmi_species = {}
        for picmi_species in self.species:
            pypicongpu_species = picmi_species.get_as_pypicongpu()
            pypicongpu_by_picmi_species[picmi_species] = pypicongpu_species
            initmgr.all_species.append(pypicongpu_species)

        # fill inter-species dependencies
        ##

        # ionization (PICMI species don't know which PyPIConGPU species they
        # use as electrons)
        self.__fill_ionization_electrons(pypicongpu_by_picmi_species)

        # operations with inter-species dependencies
        ##
        initmgr.all_operations += \
            self.__get_operations_simple_density(pypicongpu_by_picmi_species)

        # operations without inter-species dependencies
        ##
        initmgr.all_operations += \
            self.__get_operations_not_placed(pypicongpu_by_picmi_species)
        initmgr.all_operations += \
            self.__get_operations_from_individual_species(
                pypicongpu_by_picmi_species)

        return initmgr

    def __get_electron_species(self) -> PicongpuPicmiSpecies:
        """
        get electron species from existing species or generate new

        PIConGPU requires an explicit electron species, which PICMI assumes to
        implicitly already exist.
        This method retrieves an electron species by either reusing an existing
        one or generating one if missing.

        Approach:
        - 0 electron species: add one (print INFO log)
        - 1 electron species: use it
        - >1 electron species: raise, b/c is ambiguous

        electrons are identified by either mass & charge, or by particle_type.
        """
        # use caching, this is method is expensive
        if self.__electron_species is not None:
            return self.__electron_species

        all_electrons = []
        for picmi_species in self.species:
            if "electron" == picmi_species.particle_type:
                all_electrons.append(picmi_species)
            elif picmi_species.mass is not None and \
                    isclose(picmi_species.mass, constants.m_e) and \
                    picmi_species.charge is not None and \
                    isclose(picmi_species.charge, -constants.q_e):
                all_electrons.append(picmi_species)

        # exactly one electron species: use it
        if 1 == len(all_electrons):
            self.__electron_species = all_electrons[0]
            return self.__electron_species

        # no electron species: add one
        if 0 == len(all_electrons):
            # compute unambiguous name
            all_species_names = list(map(
                lambda picmi_species: picmi_species.name,
                self.species))
            electrons_name = "e"
            while electrons_name in all_species_names:
                electrons_name += "_"

            logging.info("no electron species for ionization available, "
                         "creating electrons with name: {}".format(
                             electrons_name))
            electrons = PicongpuPicmiSpecies(name=electrons_name,
                                             particle_type="electron")
            self.add_species(electrons, None)

            self.__electron_species = electrons
            return self.__electron_species

        # ambiguous choice -> raise
        raise ValueError(
            "choice of electron species for ionization is ambiguous, please "
            "set picongpu_ionization_electrons explicitly for ionizable "
            "species; found electron species: {}"
            .format(", ".join(map(lambda picmi_species: picmi_species.name,
                                  all_electrons))))

    def __resolve_electrons(self) -> None:
        """
        fill missing picongpu_ionization_electrons for ionized species

        PIConGPU needs every electron species set explicitly.
        For this, PIConGPU PICMI species have a property
        picongpu_ionization_electrons, which points to another PICMI species
        to be used for ionization.
        To be compatible to the native PICMI, this property is not required
        from the **user**, but it is stillrequired for **translation**.

        This method guesses the value of picongpu_ionization_electrons if they
        are not set.

        The actual electron selection is implemented in
        __get_electron_species()
        """
        for picmi_species in self.species:
            # only handle ionized species anyways
            if not picmi_species.has_ionizers():
                continue

            # skip if ionization electrons already set (nothing to guess)
            if picmi_species.picongpu_ionization_electrons is not None:
                continue

            picmi_species.picongpu_ionization_electrons = \
                self.__get_electron_species()

    def get_as_pypicongpu(self) -> simulation.Simulation:
        """translate to PyPIConGPU object"""
        s = simulation.Simulation()

        s.delta_t_si = self.time_step_size
        s.solver = self.solver.get_as_pypicongpu()

        if self.max_steps is not None:
            s.time_steps = self.max_steps
        elif self.max_time is not None:
            s.time_steps = self.max_time / self.time_step_size
        else:
            raise ValueError(
                "runtime not specified (neither as step count nor max time)")

        util.unsupported("verbose", self.verbose)
        util.unsupported("particle shape", self.particle_shape, "linear")
        util.unsupported("gamma boost", self.gamma_boost)

        # todo: check grid compatibility
        s.grid = self.solver.grid.get_as_pypicongpu()

        # any injection method != None is not supported
        if len(self.laser_injection_methods) != \
                self.laser_injection_methods.count(None):
            util.unsupported("laser injection method",
                             self.laser_injection_methods, [])

        if len(self.lasers) > 1:
            util.unsupported("more than one laser")

        if len(self.lasers) == 1:
            # check requires grid, so grid is translated (and thereby also
            # checked) above
            s.laser = self.lasers[0].get_as_pypicongpu()
        else:
            # explictly disable laser (as required by pypicongpu)
            s.laser = None

        # resolve electrons
        self.__resolve_electrons()

        s.init_manager = self.__get_init_manager()

        # set typical ppc if not overwritten by user
        if self.picongpu_typical_ppc is None:
            s.typical_ppc = (s.init_manager).get_typical_particle_per_cell()
        else:
            s.typical_ppc = self.picongpu_typical_ppc

        if s.typical_ppc < 1:
            raise ValueError("typical_ppc must be >= 1")

        return s

    def picongpu_run(self) -> None:
        """build and run PIConGPU simulation"""
        if self.__runner is None:
            self.__runner = runner.Runner(self.get_as_pypicongpu(),
                                          self.picongpu_template_dir)
        self.__runner.generate()
        self.__runner.build()
        self.__runner.run()

    def write_input_file(self, file_name: str) -> None:
        """
        generate input data set for picongpu

        file_name must be path to a not-yet existing directory (will be filled
        by pic-create)
        :param file_name: not yet existing directory
        """
        if self.__runner is not None:
            logging.warning("runner already initialized, overwriting")
        self.__runner = runner.Runner(self.get_as_pypicongpu(),
                                      self.picongpu_template_dir,
                                      setup_dir=file_name)
        self.__runner.generate()

    def step(self, nsteps: int = 1):
        if nsteps != self.max_steps:
            raise ValueError(
                "PIConGPU does not support stepwise running. Invoke step() "
                "with max_steps (={})".format(self.max_steps))
        self.picongpu_run()

    def picongpu_get_runner(self) -> runner.Runner:
        if self.__runner is None:
            self.__runner = runner.Runner(self.get_as_pypicongpu(),
                                          self.picongpu_template_dir)
        return self.__runner
