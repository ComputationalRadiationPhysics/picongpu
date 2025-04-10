"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from sympy import And, Eq, Piecewise
from picongpu import picmi
import numpy as np

"""
@file PICMI user script reproducing the PIConGPU bunch example
"""

time_step_size = 0.64e-17
number_of_time_steps = 7500
cellSize = np.array([0.16e-6, 0.40e-7, 0.16e-6])  # unit: meter
numberCells = np.array([128, 3072, 128])
boundary_conditions = ["periodic", "open", "periodic"]

solver = picmi.ElectromagneticSolver(
    method="Yee",
    grid=picmi.Cartesian3DGrid(
        picongpu_n_gpus=[2, 8, 2],
        number_of_cells=numberCells,
        lower_bound=[0, 0, 0],
        upper_bound=(numberCells * cellSize),
        lower_boundary_conditions=boundary_conditions,
        upper_boundary_conditions=boundary_conditions,
    ),
)


def delta_peak(x, y, z, dx, dy, dz):
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


myDensity = picmi.AnalyticDistribution(delta_peak)

base_density = 1 / cellSize.prod()

print(f"Value of my density at (1,2,3): {myDensity(1, 2, 3, 0.1, 0.2, 0.3)}")
points = np.linspace(0, 1, 10)
print(f"Values of my density with numpy arrays: {myDensity(points, points, points, 0.1, 0.2, 0.3)}")


sim = picmi.Simulation(
    solver=solver,
    max_steps=number_of_time_steps,
    time_step_size=time_step_size,
    picongpu_base_density=base_density,
)
sim.add_species(
    picmi.Species(particle_type="electron", name="electron", initial_distribution=myDensity),
    picmi.PseudoRandomLayout(n_macroparticles_per_cell=2),
)
