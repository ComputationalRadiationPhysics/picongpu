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

Evaluates the analytic electric field of a Gaussian laser on a small grid,
via the complex_amplitude / E / envelope methods of GaussianLaser.
"""

import numpy as np

from picongpu import picmi

# BEGIN-LASER-FIELDS
laser = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6,
    duration=5.0e-15,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    # put the focus (and the pulse peak at t=0) in the origin for this evaluation:
    focal_position=[0.0, 0.0, 0.0],
    centroid_position=[0.0, 0.0, 0.0],
    a0=8.0,
    phi0=0.0,
)

# a small evaluation grid around the focus, in metres:
# x/y/z must have the same shape, so build them with meshgrid
x = np.linspace(-1.0e-6, 1.0e-6, 3)
y = np.array([0.0])
z = np.array([0.0])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# the vector electric field has the components first:
# shape (3, 3, 1, 1) for the (x, y, z) components over the grid
E = laser.E(X, Y, Z, t=0.0)
print(f"E_x at the focus: {laser.Ex(0.0, 0.0, 0.0, t=0.0):.3e} V/m")

# the complex amplitude and the envelope are scalar fields
amplitude = laser.complex_amplitude(X, Y, Z, t=0.0)
envelope = laser.envelope(X, Y, Z, t=0.0)
print(f"on-axis, in-focus amplitude = {abs(amplitude[1, 0, 0]):.3e} V/m (the peak field E0)")
print(f"envelope at x = 1.0e-6 m: {envelope[2, 0, 0]:.3e} V/m")
print("It worked!")
# END-LASER-FIELDS
