"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import collections
import pdg

from scipy import constants as consts

_PropertyTuple: collections.namedtuple = collections.namedtuple("_PropertyTuple", ["mass", "charge"])

# based on 2024 Particle data Group values, @todo read automatically from somewhere, BrianMarre
_quarks = {
    "up": _PropertyTuple(
        mass=2.16e6 * consts.elementary_charge / consts.speed_of_light**2, charge=2.0 / 3.0 * consts.elementary_charge
    ),
    "charm": _PropertyTuple(
        mass=1.2730e9 * consts.elementary_charge / consts.speed_of_light**2, charge=2.0 / 3.0 * consts.elementary_charge
    ),
    "top": _PropertyTuple(
        mass=172.57e9 * consts.elementary_charge / consts.speed_of_light**2, charge=2.0 / 3.0 * consts.elementary_charge
    ),
    "down": _PropertyTuple(
        mass=4.70e6 * consts.elementary_charge / consts.speed_of_light**2, charge=-1.0 / 3.0 * consts.elementary_charge
    ),
    "strange": _PropertyTuple(
        mass=93.5 * consts.elementary_charge / consts.speed_of_light**2, charge=-1.0 / 3.0 * consts.elementary_charge
    ),
    "bottom": _PropertyTuple(
        mass=4.138 * consts.elementary_charge / consts.speed_of_light**2, charge=-1.0 / 3.0 * consts.elementary_charge
    ),
    "anti-up": _PropertyTuple(
        mass=2.16e6 * consts.elementary_charge / consts.speed_of_light**2, charge=-2.0 / 3.0 * consts.elementary_charge
    ),
    "anti-charm": _PropertyTuple(
        mass=1.2730e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=-2.0 / 3.0 * consts.elementary_charge,
    ),
    "anti-top": _PropertyTuple(
        mass=172.57e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=-2.0 / 3.0 * consts.elementary_charge,
    ),
    "anti-down": _PropertyTuple(
        mass=4.70e6 * consts.elementary_charge / consts.speed_of_light**2, charge=1.0 / 3.0 * consts.elementary_charge
    ),
    "anti-strange": _PropertyTuple(
        mass=93.5 * consts.elementary_charge / consts.speed_of_light**2, charge=1.0 / 3.0 * consts.elementary_charge
    ),
    "anti-bottom": _PropertyTuple(
        mass=4.138 * consts.elementary_charge / consts.speed_of_light**2, charge=1.0 / 3.0 * consts.elementary_charge
    ),
}

_leptons = {
    "electron": _PropertyTuple(mass=consts.electron_mass, charge=-consts.elementary_charge),
    "muon": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("mu-").mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("mu-").charge * consts.elementary_charge,
    ),
    "tau": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("tau-").mass
        * 1e9
        * consts.elementary_charge
        / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("tau-").charge * consts.elementary_charge,
    ),
    "positron": _PropertyTuple(mass=consts.electron_mass, charge=consts.elementary_charge),
    "anti-muon": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("mu+").mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("mu+").charge * consts.elementary_charge,
    ),
    "anti-tau": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("tau+").mass
        * 1e9
        * consts.elementary_charge
        / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("tau+").charge * consts.elementary_charge,
    ),
}

_nucleons = {
    "proton": _PropertyTuple(mass=consts.proton_mass, charge=consts.elementary_charge),
    "anti-proton": _PropertyTuple(mass=consts.proton_mass, charge=-consts.elementary_charge),
    "neutron": _PropertyTuple(mass=consts.neutron_mass, charge=None),
    "anti-neutron": _PropertyTuple(mass=consts.neutron_mass, charge=None),
}

_neutrinos = {
    "electron-neutrino": _PropertyTuple(mass=0.0, charge=0.0),
    "muon-neutrino": _PropertyTuple(mass=0.0, charge=0.0),
    "tau-neutrino": _PropertyTuple(mass=0.0, charge=0.0),
    "anti-electron-neutrino": _PropertyTuple(mass=0.0, charge=0.0),
    "anti-muon-neutrino": _PropertyTuple(mass=0.0, charge=0.0),
    "anti-tau-neutrino": _PropertyTuple(mass=0.0, charge=0.0),
}

_gauge_bosons = {
    "photon": _PropertyTuple(mass=None, charge=0.0),
    "gluon": _PropertyTuple(mass=None, charge=0.0),
    "w-plus-boson": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("W+").mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("W+").charge * consts.elementary_charge,
    ),
    "w-minus-boson": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("W-").mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("W-").charge * consts.elementary_charge,
    ),
    "z-boson": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("Z").mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("Z").charge * consts.elementary_charge,
    ),
    "higgs": _PropertyTuple(
        mass=pdg.connect().get_particle_by_name("H").mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=pdg.connect().get_particle_by_name("H").charge * consts.elementary_charge,
    ),
}

non_element_particle_type_properties = {**_quarks, **_leptons, **_neutrinos, **_nucleons, **_gauge_bosons}
