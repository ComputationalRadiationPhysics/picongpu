"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ..pypicongpu import util, species

import picmistandard

from typeguard import typechecked, check_type
import typing
import logging
from scipy import constants as consts


@typechecked
class Species(picmistandard.PICMI_Species):
    """PICMI object for a (single) particle species"""

    # ONLY set non-element particles here -- all other are handled by
    # element
    __mass_charge_by_openpmd_name_non_elements = {
        "electron": (consts.electron_mass, -consts.elementary_charge),
    }
    """mass/charge tuple to use when passed a non-element particle_type"""

    picongpu_fully_ionized = \
        util.build_typesafe_property(typing.Optional[bool])
    """
    *usually* ionization is expected to be used on elements -- use this to
    explicitly DISABLE ionization
    """

    def __init__(
            self,
            picongpu_fully_ionized: typing.Optional[bool] = None,
            picongpu_ionization_electrons=None,
            **kw):
        self.picongpu_fully_ionized = picongpu_fully_ionized

        # note: picongpu_ionization_electrons would *normally* just use a
        # forward-declared typecheck "Species"
        # However, this requires that "Species" at some point resolves to this
        # class. Typically this picmi species object is only available as
        # "picmi.Species()", and the resolution fails.
        # Hence, the type is checked manually here.
        check_type("picongpu_ionization_electrons",
                   picongpu_ionization_electrons,
                   typing.Optional[Species])
        self.picongpu_ionization_electrons = picongpu_ionization_electrons

        super().__init__(**kw)

    @staticmethod
    def __get_temperature_kev_by_rms_velocity(
            rms_velocity_si: typing.Tuple[float, float, float],
            particle_mass_si: float) -> float:
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
        assert rms_velocity_si[0] == rms_velocity_si[1] and \
            rms_velocity_si[1] == rms_velocity_si[2], \
            "all thermal velcoity spread (rms velocity) components must be " \
            "equal"
        # see
        # https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
        rms_velocity_si_squared = rms_velocity_si[0] ** 2
        return particle_mass_si * rms_velocity_si_squared \
            * consts.electron_volt ** -1 * 10 ** -3

    def __get_drift(
            self) -> typing.Optional[species.operation.momentum.Drift]:
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
        check if particle type is set, if yes set self.mass and self.charge
        """
        if self.particle_type is None:
            return

        # particle type is set -> retrieve mass & charge
        assert self.charge is None, \
            "charge is specify implicitly via particle type, " \
            "do NOT set charge explictly"
        assert self.mass is None, \
            "mass is specify implicitly via particle type, " \
            "do NOT set mass explictly"

        if self.particle_type in \
                self.__mass_charge_by_openpmd_name_non_elements:
            # not element, but known
            mass_charge_tuple = \
                self.__mass_charge_by_openpmd_name_non_elements[
                    self.particle_type]
            self.mass = mass_charge_tuple[0]
            self.charge = mass_charge_tuple[1]
        else:
            # element (or unkown, which raises when trying to get an
            # element for that name)
            self.element = species.util.Element.get_by_openpmd_name(
                self.particle_type)
            self.mass = self.element.get_mass_si()
            self.charge = self.element.get_charge_si()

    def __check_ionization(self) -> None:
        """
        check if ionization (charge_state) can be applied, potentially warns
        """
        assert not self.picongpu_fully_ionized \
            or self.charge_state is None, \
            "picongpu_fully_ionized may only be used if " \
            "charge_state is none"

        if self.particle_type is None:
            # no particle type -> charge state is not allowed
            assert self.charge_state is None, "charge_state is ONLY allowed " \
                "when setting particle_type explicitly"

            # no particle type -> fully ionized flag not permitted
            assert self.picongpu_fully_ionized is None, \
                "picongpu_fully_ionized is ONLY allowed " \
                "when setting particle_type explicitly"

            # no charge_state -> nothing left
            return

        # particle type is set: fully ionized flag *ONLY* allowed if using
        # element
        if self.picongpu_fully_ionized is not None:
            assert self.particle_type not in \
                self.__mass_charge_by_openpmd_name_non_elements, \
                "picongpu_fully_ionized is ONLY allowed for elements"

        # maybe warn
        if self.charge_state is None:
            # theoretically speaking atoms *always* have a charge state
            # for PIConGPU an atom (ion) may exist without a charge state,
            # i.e. without ionization, however this may result in
            # (physically) incorrect behavior
            # Therefore warn if there is no charge state -- unless this
            # warning is explicitly disabled with a flag is given

            # (note: omit if not element)
            if not self.picongpu_fully_ionized and \
                    self.particle_type not in \
                    self.__mass_charge_by_openpmd_name_non_elements:
                logging.warning(
                    "species {} will be fully ionized for the entire "
                    "simulation -- if this is intended, set "
                    "picongpu_fully_ionized=True"
                    .format(self.name))

    def get_as_pypicongpu(self) -> species.Species:
        util.unsupported("method", self.method)
        util.unsupported("particle shape", self.particle_shape)
        # note: placement params are respected in associated simulation object

        assert self.name is not None, "name must be set"

        self.__maybe_apply_particle_type()
        self.__check_ionization()

        s = species.Species()
        s.name = self.name
        s.constants = []

        if self.mass:
            # if 0==mass rather omit mass entirely
            assert self.mass > 0

            mass_constant = species.constant.Mass()
            mass_constant.mass_si = self.mass
            s.constants.append(mass_constant)

        if self.charge is not None:
            charge_constant = species.constant.Charge()
            charge_constant.charge_si = self.charge
            s.constants.append(charge_constant)

        if self.density_scale is not None:
            assert self.density_scale > 0

            density_scale_constant = species.constant.DensityRatio()
            density_scale_constant.ratio = self.density_scale
            s.constants.append(density_scale_constant)

        if self.particle_type and \
           self.particle_type not in \
           self.__mass_charge_by_openpmd_name_non_elements:
            # particle type given and is not non-element (==is element)
            # -> add element flags
            element = \
                species.util.Element.get_by_openpmd_name(self.particle_type)

            elementary_properties_const = species.constant.ElementProperties()
            elementary_properties_const.element = element
            s.constants.append(elementary_properties_const)

        if self.charge_state is not None:
            # element must be set from previous code section
            assert element is not None

            atomic_number = element.value
            assert self.charge_state <= atomic_number, \
                "charge_state must be <= atomic number ({})" \
                .format(atomic_number)

            const_ionizers = species.constant.Ionizers()
            # const_ionizers.electron_species must be set to a pypicongpu
            # species, but this is not available here
            # -> inserted externally
            s.constants.append(const_ionizers)

        return s

    def has_ionizers(self) -> bool:
        """
        returns true iff a species will have ionizers (algorithms)
        """
        return self.charge_state is not None

    def get_independent_operations(
            self, pypicongpu_species: species.Species) \
            -> typing.List[species.operation.Operation]:
        assert pypicongpu_species.name == self.name, "to generate " \
            "operations for PyPIConGPU species: names must match"

        all_operations = []

        # assign momentum
        momentum_op = species.operation.SimpleMomentum()
        momentum_op.species = pypicongpu_species
        momentum_op.drift = self.__get_drift()

        temperature_kev = 0
        if self.initial_distribution is not None and \
           self.initial_distribution.rms_velocity is not None:
            mass_const = pypicongpu_species.get_constant_by_type(
                species.constant.Mass)
            mass_si = mass_const.mass_si

            temperature_kev = \
                self.__get_temperature_kev_by_rms_velocity(
                    tuple(self.initial_distribution.rms_velocity),
                    mass_si)

        if 0 != temperature_kev:
            momentum_op.temperature = \
                species.operation.momentum.Temperature()
            momentum_op.temperature.temperature_kev = temperature_kev
        else:
            momentum_op.temperature = None

        all_operations.append(momentum_op)

        # ionization:
        if self.has_ionizers():
            # note: this will raise if called *before* get_as_pypicongpu with
            # "self.element" is not defined -- in this case, either fix the
            # order or compute the element here on the fly
            atomic_number = self.element.value

            # fully ionized?
            if self.charge_state == atomic_number:
                ion_op_no_electrons = species.operation.NoBoundElectrons()
                ion_op_no_electrons.species = pypicongpu_species
                all_operations.append(ion_op_no_electrons)
            else:
                # not fully ionized
                bound_electrons = atomic_number - self.charge_state
                assert bound_electrons > 0

                ion_op_electrons = species.operation.SetBoundElectrons()
                ion_op_electrons.species = pypicongpu_species
                ion_op_electrons.bound_electrons = bound_electrons

                all_operations.append(ion_op_electrons)

        return all_operations
