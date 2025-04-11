"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from scipy.constants import c
from sympy import And, Eq, Piecewise
from picongpu import picmi
import numpy as np

"""
@file PICMI user script reproducing the PIConGPU bunch example
"""

OUTPUT_DIRECTORY_PATH = "bunch"

time_step_size = 0.64e-17
number_of_time_steps = 7500
cell_size = np.array([0.16e-6, 0.40e-7, 0.16e-6])  # unit: meter
number_cells = np.array([128, 3072, 128])
distinguished_cell = [1.024e-5, 9.072e-5, 1.024e-5] // cell_size
boundary_conditions = ["periodic", "open", "periodic"]

solver = picmi.ElectromagneticSolver(
    method="Yee",
    grid=picmi.Cartesian3DGrid(
        picongpu_n_gpus=[2, 8, 2],
        number_of_cells=number_cells,
        lower_bound=[0, 0, 0],
        upper_bound=(number_cells * cell_size),
        lower_boundary_conditions=boundary_conditions,
        upper_boundary_conditions=boundary_conditions,
    ),
)


def delta_peak(x, y, z):
    current_cell_x = x // cell_size[0]
    current_cell_y = y // cell_size[1]
    current_cell_z = z // cell_size[2]

    return Piecewise(
        (
            1.0,
            And(
                Eq(current_cell_x, distinguished_cell[0]),
                Eq(current_cell_y, distinguished_cell[1]),
                Eq(current_cell_z, distinguished_cell[2]),
            ),
        ),
        (0.0, True),
    )


# A shorter version in arbitrary dimensions would be the following.
# But it's probably less clear to read for a physicist.
def delta_peak_short(*position):
    return Piecewise(
        (
            1.0,
            And(*(Eq(x // s, d) for x, s, d in zip(position, cell_size, distinguished_cell))),
        ),
        (0.0, True),
    )


def velocity(gamma):
    return np.sqrt(c**2 * (1.0 - 1.0 / gamma**2))


myDensity = picmi.AnalyticDistribution(delta_peak, directed_velocity=-velocity(gamma=5.0) * np.eye(3)[1, :])

base_density = 1 / cell_size.prod()

print(f"Value of my density at (1,2,3): {myDensity(1, 2, 3)}")
points = np.linspace(0, 1, 10)
print(f"Values of my density with numpy arrays: {myDensity(points, points, points)}")


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

if __name__ == "__main__":
    sim.write_input_file(OUTPUT_DIRECTORY_PATH)
