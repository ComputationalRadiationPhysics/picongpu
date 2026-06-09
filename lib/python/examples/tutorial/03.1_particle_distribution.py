#!/usr/bin/env -S uv run

# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib",
#   "mpl_ascii",
#   "numpy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python",
#   "sympy"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from picongpu.picmi.distribution import GaussianDistribution, AnalyticDistribution
from sympy import Piecewise, Abs, exp

NUM_CELLS = np.array([192, 2048, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])

particle_distribution = GaussianDistribution(
    density=1.0e25,
    center_front=8.0e-5,
    sigma_front=8.0e-5,
    center_rear=10.0e-5,
    sigma_rear=8.0e-5,
    factor=-1.0,
    power=4.0,
    vacuum_front=50 * CELL_SIZE[1],
    # temporary workaround:
    cell_size=CELL_SIZE,
)

x = NUM_CELLS[0] / 2 * CELL_SIZE[0]
y, z = np.mgrid[: 2 * NUM_CELLS[1], : NUM_CELLS[2]] * CELL_SIZE[1:3, np.newaxis, np.newaxis]
predefined_values = particle_distribution(x, y, z)

matplotlib.use("module://mpl_ascii")
plt.figure()
plt.contour(y, z, predefined_values)


@AnalyticDistribution
def custom_particle_distribution(x, y, z):
    # PIConGPU's predefined GaussianDistribution is evaluated at the center of the cell
    y += -0.5 * CELL_SIZE[1]
    # The last term undoes the shift to the cell origin.
    vacuum_y = int(particle_distribution.vacuum_front / CELL_SIZE[1]) * CELL_SIZE[1] - 0.5 * CELL_SIZE[1]

    exponent = Piecewise(
        (
            Abs((y - particle_distribution.center_front) / particle_distribution.sigma_front),
            y < particle_distribution.center_front,
        ),
        (
            Abs((y - particle_distribution.center_rear) / particle_distribution.sigma_rear),
            y >= particle_distribution.center_rear,
        ),
        (0.0, True),
    )
    return Piecewise(
        (0.0, y < vacuum_y),
        (
            particle_distribution.density * exp(particle_distribution.factor * exponent**particle_distribution.power),
            True,
        ),
    )


custom_values = custom_particle_distribution(x, y, z)
plt.figure()
plt.contour(y, z, custom_values)
plt.show()
