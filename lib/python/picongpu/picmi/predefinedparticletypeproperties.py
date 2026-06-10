"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import collections
import particle

from pydantic import BaseModel
import typeguard

from scipy import constants as consts

PropertyTuple: collections.namedtuple = collections.namedtuple("_PropertyTuple", ["mass", "charge"])


class PredefinedParticleTypeProperties(BaseModel):
    _particle_type_to_pdgid: dict[str, int] = {
        "down": 1,
        "up": 2,
        "strange": 3,
        "charm": 4,
        "bottom": 5,
        "top": 6,
        "anti-down": -1,
        "anti-up": -2,
        "anti-charm": -4,
        "anti-top": -6,
        "anti-strange": -3,
        "anti-bottom": -5,
        "electron": 11,
        "positron": -11,
        "muon": 13,
        "anti-muon": -13,
        "tau": 15,
        "anti-tau": -15,
        "gluon": 21,
        "photon": 22,
        "z-boson": 23,
        "w-plus-boson": 24,
        "w-minus-boson": -24,
        "higgs": 25,
    }

    _directDefinitions: dict[str, PropertyTuple] = {
        "proton": PropertyTuple(mass=consts.proton_mass, charge=consts.elementary_charge),
        "anti-proton": PropertyTuple(mass=consts.proton_mass, charge=-consts.elementary_charge),
        "neutron": PropertyTuple(mass=consts.neutron_mass, charge=None),
        "anti-neutron": PropertyTuple(mass=consts.neutron_mass, charge=None),
    }

    def get_known_particle_types(self) -> list[str]:
        return list(self._directDefinitions.keys()) + list(self._particle_type_to_pdgid.keys())

    @typeguard.typechecked
    def get_mass_and_charge_of_non_element(self, particle_type: str) -> PropertyTuple:
        """mass and charge of physical particle of specified non element particle type

        @param particle_type as defined in the openPMD upcoming-2.0.0 extension SpeciesType
        @detail based on Particle data Group(pdg) values of installed instance of python package particle

        @returns None if particle_type is unknown, units: (kg, C)
        """

        if particle_type in self._particle_type_to_pdgid.keys():
            data = particle.Particle.from_pdgid(self._particle_type_to_pdgid[particle_type])
            propertyTuple = PropertyTuple(
                mass=data.mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
                charge=data.charge * consts.elementary_charge,
            )

        elif particle_type in self._directDefinitions.keys():
            propertyTuple = self._directDefinitions[particle_type]

        else:
            return None

        # replace charge 0 or mass 0 with None
        if propertyTuple.mass == 0.0:
            propertyTuple = PropertyTuple(mass=None, charge=propertyTuple.charge)
        if propertyTuple.charge == 0.0:
            propertyTuple = PropertyTuple(mass=propertyTuple.mass, charge=None)

        return propertyTuple
