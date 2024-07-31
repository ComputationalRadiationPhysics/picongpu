"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .predefinedparticletypeproperties import non_element_particle_type_properties
from .interaction import Interaction

from .. import pypicongpu
from ..pypicongpu.species.util.element import Element

import picmistandard

import typing
import typeguard
import logging
import re

from scipy import constants as consts


@typeguard.typechecked
class Species(picmistandard.PICMI_Species):
    """PICMI object for a (single) particle species"""

    __non_element_particle_type_properties = non_element_particle_type_properties
    """
    mass/charge to use when passed a non-element particle_type

    @attention ONLY set non-element particles here, all other are handled by element
    """

    picongpu_element = pypicongpu.util.build_typesafe_property(typing.Optional[Element])
    """element information of object"""

    __non_element_particle_types: list[str] = __non_element_particle_type_properties.keys()
    """list of particle types"""

    picongpu_fixed_charge = pypicongpu.util.build_typesafe_property(bool)

    interactions = pypicongpu.util.build_typesafe_property(typing.Optional[list[None]])
    """overwrite base class interactions to disallow setting them"""

    __warned_already: bool = False
    __previous_check: bool = False

    def __init__(self, picongpu_fixed_charge: bool = False, **keyword_arguments):
        self.picongpu_fixed_charge = picongpu_fixed_charge
        self.picongpu_element = None

        # let PICMI class handle remaining init
        picmistandard.PICMI_Species.__init__(self, **keyword_arguments)

    @staticmethod
    def __get_temperature_kev_by_rms_velocity(
        rms_velocity_si: tuple[float, float, float], particle_mass_si: float
    ) -> float:
        """
        convert temperature from RMS velocity vector to keV

        Uses assertions to reject incorrect format.
        Ensures that all three vector components are equal and >0.

        Helper function invoked from inside Distribution classes.

        :param rms_velocity_si: rms velocity (thermal velocity spread) per
                                direction in m/s
        :param particle_mass_si: particle mass in kg
        :raises Exception: on impossible conversion
        :return: temperature in keV
        """
        assert rms_velocity_si[0] == rms_velocity_si[1] and rms_velocity_si[1] == rms_velocity_si[2], (
            "all thermal velcoity spread (rms velocity) components must be " "equal"
        )
        # see
        # https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
        rms_velocity_si_squared = rms_velocity_si[0] ** 2
        return particle_mass_si * rms_velocity_si_squared * consts.electron_volt**-1 * 10**-3

    def __get_drift(self) -> typing.Optional[pypicongpu.species.operation.momentum.Drift]:
        """
        Retrieve respective pypicongpu drift object (or None)

        Returns none if: rms_velocity is 0 OR distribution is not set

        :return: drift object (might be none)
        """
        if self.initial_distribution is None:
            return None
        return self.initial_distribution.get_picongpu_drift()

    def __maybe_apply_particle_type(self) -> None:
        """
        if particle type is set, set self.mass, self.charge and element from particle type

        necessary to ensure consistent state regardless which parameters the user specified in species init

        @raises if both particle_type and charge mass are specified
        """

        if (self.particle_type is None) or re.match(r"other:.*", self.particle_type):
            # no particle or custom particle type set
            pass
        else:
            # set mass & charge
            if self.particle_type in self.__non_element_particle_types:
                # not element, but known
                mass_charge_tuple = self.__non_element_particle_type_properties[self.particle_type]

                self.mass = mass_charge_tuple.mass
                self.charge = mass_charge_tuple.charge
            elif Element.is_element(self.particle_type):
                # element or similar, will raise if element name is unknown
                self.picongpu_element = pypicongpu.species.util.Element(self.particle_type)
                self.mass = self.picongpu_element.get_mass_si()
                self.charge = self.picongpu_element.get_charge_si()
            else:
                # unknown particle type
                raise ValueError(f"Species {self.name} has unknown particle type {self.particle_type}")

    def has_ionization(self, interaction: Interaction | None) -> bool:
        """does species have ionization configured?"""
        if interaction is None:
            return False
        if interaction.has_ionization(self):
            return True

        # to get typecheck to shut up
        return False

    def is_ion(self) -> bool:
        """
        is species an ion?

        @attention requires __maybe_apply_particle_type() to have been called first,
            otherwise will return wrong result
        """
        if self.picongpu_element is None:
            return False
        return True

    def __check_ionization_configuration(self, interaction: Interaction | None) -> None:
        """
        check species ioniaztion- and species- configuration are compatible

        @raises if incorrect configuration found
        """

        if self.particle_type is None:
            assert not self.has_ionization(
                interaction
            ), f"Species {self.name} configured with active ionization but required particle_type not set."
            assert self.charge_state is None, (
                f"Species {self.name} specified initial charge state via charge_state without also specifying particle "
                "type, must either set particle_type explicitly or only use charge instead"
            )
            assert (
                self.picongpu_fixed_charge is False
            ), f"Species {self.name} specified fixed charge without also specifying particle_type"
        else:
            # particle type is
            if (self.particle_type in self.__non_element_particle_types) or re.match(r"other:.*", self.particle_type):
                # non ion predefined particle, or custom particle type
                assert self.charge_state is None, "charge_state may only be set for ions"
                assert not self.has_ionization(
                    interaction
                ), f"Species {self.name} configured with active ionization but particle type indicates non ion."
                assert (
                    self.picongpu_fixed_charge is False
                ), f"Species {self.name} configured with fixed charge state but particle_type indicates non ion"
            elif Element.is_element(self.particle_type):
                # ion

                # check for unphysical charge state
                if self.charge_state is not None:
                    assert (
                        Element(self.particle_type).get_atomic_number() >= self.charge_state
                    ), f"Species {self.name} intial charge state is unphysical"

                if self.has_ionization(interaction):
                    assert self.picongpu_fixed_charge is False, (
                        f"Species {self.name} configured both as fixed charge ion and ion with ionization, may be "
                        " either or but not both."
                    )
                    assert self.charge_state is not None, (
                        f"Species {self.name} configured with ionization but no initial charge state specified, "
                        "must be explicitly specified via charge_state."
                    )
                else:
                    # ion with fixed charge
                    if self.picongpu_fixed_charge is False:
                        raise ValueError(
                            f"Species {self.name} configured with fixed charge state without explicitly setting picongpu_fixed_charge=True"
                        )

                    if not self.__warned_already:
                        logging.warning(
                            f"Species {self.name} configured with fixed charge state but particle type"
                            "indicates element. This is not recommended but supported"
                        )
                        self.__warned_already = True

                    # charge_state may be set or None indicating some fixed number of bound electrons or fully ionized
                    #   ion
            else:
                # unknown particle type
                raise ValueError(f"unknown particle type {self.particle_type} in species {self.name}")

    def __check_interaction_configuration(self, interaction: Interaction | None) -> None:
        """check all interactions sub groups for compatibility with this species configuration"""
        self.__check_ionization_configuration(interaction)

    def check(self, interaction: Interaction | None) -> None:
        assert self.name is not None, "picongpu requires each species to have a name set."

        # check charge and mass explicitly set/not set depending on particle_type
        if (self.particle_type is None) or re.match(r"other:.*", self.particle_type):
            # custom species may not have mass or charge
            pass
        elif not self.__previous_check:
            assert (
                self.charge is None
            ), f"Species' {self.name}, charge is specified implicitly via particle type, do NOT set charge explictly"
            assert (
                self.mass is None
            ), f"Species' {self.name}, mass is specified implicitly via particle type, do NOT set mass explictly"

        self.__check_interaction_configuration(interaction)
        self.__previous_check = True

    def get_as_pypicongpu(
        self, interaction: Interaction | None
    ) -> tuple[
        pypicongpu.species.Species, None | dict[typing.Any, pypicongpu.species.constant.ionizationmodel.IonizationModel]
    ]:
        """
        translate PICMI species object to equivalent PyPIConGPU species object

        @attention only translates ONLY species owned objects, for example species-Constants
            everything else requires a call to the corresponding getter of this class
        """

        # error on unsupported options
        pypicongpu.util.unsupported("method", self.method)
        pypicongpu.util.unsupported("particle shape", self.particle_shape)
        # @note placement params are respected in associated simulation object

        self.check(interaction)
        self.__maybe_apply_particle_type()

        s = pypicongpu.species.Species()
        s.name = self.name
        s.constants = []

        if self.mass:
            # if 0==mass rather omit mass entirely
            assert self.mass > 0

            mass_constant = pypicongpu.species.constant.Mass()
            mass_constant.mass_si = self.mass
            s.constants.append(mass_constant)

        if self.density_scale is not None:
            assert self.density_scale > 0

            density_scale_constant = pypicongpu.species.constant.DensityRatio()
            density_scale_constant.ratio = self.density_scale
            s.constants.append(density_scale_constant)

        # default case species with no charge and/or no bound electrons or with ionization
        charge_constant_value = self.charge

        initial_charge_state_set = self.charge_state is not None
        fixed_charge_state = not self.has_ionization(interaction)
        if self.is_ion() and initial_charge_state_set and fixed_charge_state:
            # fixed not completely ionized ion
            charge_constant_value = self.charge_state * consts.elementary_charge

        if charge_constant_value is not None:
            charge_constant = pypicongpu.species.constant.Charge()
            charge_constant.charge_si = charge_constant_value
            s.constants.append(charge_constant)

        if interaction is not None:
            interaction_constants, pypicongpu_model_by_picmi_model = interaction.get_interaction_constants(self)
            s.constants.extend(interaction_constants)
        else:
            pypicongpu_model_by_picmi_model = None

        return s, pypicongpu_model_by_picmi_model

    def get_independent_operations(
        self, pypicongpu_species: pypicongpu.species.Species, interaction: Interaction | None
    ) -> list[pypicongpu.species.operation.Operation]:
        """get a list of all operations only initializing attributes of this species"""

        # assure consistent state of species
        self.check(interaction)
        self.__maybe_apply_particle_type()

        assert pypicongpu_species.name == self.name, (
            "to generate " "operations for PyPIConGPU species: names must match"
        )

        all_operations = []

        # assign momentum
        momentum_op = pypicongpu.species.operation.SimpleMomentum()
        momentum_op.species = pypicongpu_species
        momentum_op.drift = self.__get_drift()

        temperature_kev = 0
        if self.initial_distribution is not None and self.initial_distribution.rms_velocity is not None:
            mass_const = pypicongpu_species.get_constant_by_type(pypicongpu.species.constant.Mass)
            mass_si = mass_const.mass_si

            temperature_kev = self.__get_temperature_kev_by_rms_velocity(
                tuple(self.initial_distribution.rms_velocity), mass_si
            )

        if 0 != temperature_kev:
            momentum_op.temperature = pypicongpu.species.operation.momentum.Temperature()
            momentum_op.temperature.temperature_kev = temperature_kev
        else:
            momentum_op.temperature = None

        all_operations.append(momentum_op)

        # assign boundElectrons attribute
        if self.is_ion() and self.has_ionization(interaction):
            bound_electrons_op = pypicongpu.species.operation.SetChargeState()
            bound_electrons_op.species = pypicongpu_species
            bound_electrons_op.charge_state = self.charge_state
            all_operations.append(bound_electrons_op)
        else:
            # fixed charge state -> therefore no bound electron attribute necessary
            pass

        return all_operations
