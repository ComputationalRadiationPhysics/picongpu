"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from sympy import And, Eq, Piecewise, KroneckerDelta, tanh
from picongpu import picmi
import numpy as np

"""
@file PICMI user script reproducing the PIConGPU bunch example
"""


def myDeltaPeak(x, y, z, dx, dy, dz):
    x0 = 1.024e-5
    y0 = 9.072e-5
    z0 = 1.024e-5

    id_x = x // dx
    id_y = y // dy
    id_z = z // dz

    id_x0 = x0 // dx
    id_y0 = y0 // dy
    id_z0 = z0 // dz

    return Piecewise((1.0, And(Eq(id_x, id_x0), Eq(id_y, id_y0), Eq(id_z, id_z0))), (0.0, True))


def myDeltaPeak2(x, y, z, dx, dy, dz):
    x0 = 1.024e-5
    y0 = 9.072e-5
    z0 = 1.024e-5

    id_x = x // dx
    id_y = y // dy
    id_z = z // dz

    id_x0 = x0 // dx
    id_y0 = y0 // dy
    id_z0 = z0 // dz

    # This cannot be printed currently.
    # We'd have to define a translation of
    # `KroneckerDelta` into C++/alpaka code ourselves.
    # Seems quite doable but doesn't exist currently.
    return KroneckerDelta(id_x, id_x0) * KroneckerDelta(id_y, id_y0) * KroneckerDelta(id_z, id_z0)


def profile_from_picmi_standard(x, y, z, dx, dy, dz):
    return 1.0e23 * (1 + tanh((z - 20.0e-6) / 10.0e-6)) / 2.0


myDensity = picmi.AnalyticDistribution(profile_from_picmi_standard)
myDensity2 = picmi.AnalyticDistribution(myDeltaPeak)

print(f"Value of my density at (1,2,3): {myDensity(1, 2, 3, 0.1, 0.2, 0.3)}")
print(f"Value of my second density at (1,2,3): {myDensity2(1, 2, 3, 0.1, 0.2, 0.3)}")
points = np.linspace(0, 1, 10)
print(f"Values of my density with numpy arrays: {myDensity(points, points, points, 0.1, 0.2, 0.3)}")
print(f"Values of my second density with numpy arrays: {myDensity2(points, points, points, 0.1, 0.2, 0.3)}")


numberCells = np.array([192, 2048, 192])
cellSize = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])  # unit: meter
grid = picmi.Cartesian3DGrid(
    picongpu_n_gpus=[2, 4, 1],
    number_of_cells=numberCells.tolist(),
    lower_bound=[0, 0, 0],
    upper_bound=(numberCells * cellSize).tolist(),
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")

sim = picmi.Simulation(
    solver=solver,
    max_steps=4000,
    time_step_size=1.39e-16,
    picongpu_moving_window_move_point=0.9,
)
sim.add_species(
    picmi.Species(particle_type="electron", name="electron", initial_distribution=myDensity),
    picmi.PseudoRandomLayout(n_macroparticles_per_cell=2),
)
sim.add_species(
    picmi.Species(particle_type="electron", name="electron_delta", initial_distribution=myDensity2),
    picmi.PseudoRandomLayout(n_macroparticles_per_cell=2),
)
