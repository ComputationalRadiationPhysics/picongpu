"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import collections
import particle

from scipy import constants as consts

_PropertyTuple: collections.namedtuple = collections.namedtuple("_PropertyTuple", ["mass", "charge"])

# based on 2024 Particle data Group values, @todo read automatically from somewhere, BrianMarre
_quarks = {
    "up": _PropertyTuple(
        mass=particle.Particle.findall("u")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("u")[0].charge * consts.elementary_charge,
    ),
    "charm": _PropertyTuple(
        mass=particle.Particle.findall("c")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("c")[0].charge * consts.elementary_charge,
    ),
    "top": _PropertyTuple(
        mass=particle.Particle.findall("t")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("t")[0].charge * consts.elementary_charge,
    ),
    "down": _PropertyTuple(
        mass=particle.Particle.findall("d")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("d")[0].charge * consts.elementary_charge,
    ),
    "strange": _PropertyTuple(
        mass=particle.Particle.findall("s")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("s")[0].charge * consts.elementary_charge,
    ),
    "bottom": _PropertyTuple(
        mass=particle.Particle.findall("b")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("b")[0].charge * consts.elementary_charge,
    ),
    "anti-up": _PropertyTuple(
        mass=particle.Particle.findall("u~")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("u~")[0].charge * consts.elementary_charge,
    ),
    "anti-charm": _PropertyTuple(
        mass=particle.Particle.findall("c~")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("c~")[0].charge * consts.elementary_charge,
    ),
    "anti-top": _PropertyTuple(
        mass=particle.Particle.findall("t~")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("t~")[0].charge * consts.elementary_charge,
    ),
    "anti-down": _PropertyTuple(
        mass=particle.Particle.findall("d~")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("d~")[0].charge * consts.elementary_charge,
    ),
    "anti-strange": _PropertyTuple(
        mass=particle.Particle.findall("s~")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("s~")[0].charge * consts.elementary_charge,
    ),
    "anti-bottom": _PropertyTuple(
        mass=particle.Particle.findall("b~")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("b~")[0].charge * consts.elementary_charge,
    ),
}

_leptons = {
    "electron": _PropertyTuple(mass=consts.electron_mass, charge=-consts.elementary_charge),
    "muon": _PropertyTuple(
        mass=particle.Particle.findall("mu-")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("mu-")[0].charge * consts.elementary_charge,
    ),
    "tau": _PropertyTuple(
        mass=particle.Particle.findall("tau-")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("tau-")[0].charge * consts.elementary_charge,
    ),
    "positron": _PropertyTuple(mass=consts.electron_mass, charge=consts.elementary_charge),
    "anti-muon": _PropertyTuple(
        mass=particle.Particle.findall("mu+")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("mu+")[0].charge * consts.elementary_charge,
    ),
    "anti-tau": _PropertyTuple(
        mass=particle.Particle.findall("tau+")[0].mass * 1e6 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("tau+")[0].charge * consts.elementary_charge,
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
        mass=particle.Particle.findall("W+")[0].mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("W+")[0].charge * consts.elementary_charge,
    ),
    "w-minus-boson": _PropertyTuple(
        mass=particle.Particle.findall("W-")[0].mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("W-")[0].charge * consts.elementary_charge,
    ),
    "z-boson": _PropertyTuple(
        mass=particle.Particle.findall("Z")[0].mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("Z")[0].charge * consts.elementary_charge,
    ),
    "higgs": _PropertyTuple(
        mass=particle.Particle.findall("H")[0].mass * 1e9 * consts.elementary_charge / consts.speed_of_light**2,
        charge=particle.Particle.findall("H")[0].charge * consts.elementary_charge,
    ),
}

non_element_particle_type_properties = {**_quarks, **_leptons, **_neutrinos, **_nucleons, **_gauge_bosons}
