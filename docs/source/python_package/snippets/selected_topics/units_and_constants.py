#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Converts between common units and the SI values the frontend expects,
using the physical constants that ship with the PICMI frontend.
"""

from math import pi

from picongpu.picmi import constants

# energy: the frontend takes joules; constants.keV is one kiloelectronvolt in J
max_energy_si = 500.0 * constants.keV

# momentum: the frontend takes SI momenta (kg m/s);
# one electron rest-mass momentum is m_e * c
one_rest_mass_momentum_si = constants.m_e * constants.c

# the constants are also usable in derived expressions,
# e.g. the electric field amplitude E0 corresponding to a normalized
# vector potential a0 and wavenumber k0 (cf. the lasers page):
a0 = 8.0
wavelength = 0.8e-6
k0 = 2.0 * pi / wavelength
E0_si = a0 * constants.m_e * constants.c**2 * k0 / constants.q_e

print(f"max_energy_si = {max_energy_si:.3e} J")
print(f"one_rest_mass_momentum_si = {one_rest_mass_momentum_si:.3e} kg m/s")
print(f"E0_si = {E0_si:.3e} V/m")
print("It worked!")
