#!/usr/bin/env python

"""
# SPDX-FileCopyrightText: Marco Garten, Pawel Ordyna,Brian Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from picongpu.extra.utils.memory_calculator import MemoryCalculator

import numpy as np

"""
@file

This file contains a usage example of the ``MemoryCalculator`` class
for our :ref:`FoilLCT example <usage-examples-foilLCT>` and its ``4.cfg``.

It calculates an estimate for how much memory is used per device if the whole
target would be fully ionized but does not move much.

The no-movement approximation may severely underestimate the memory requirements
of individual GPUs, especially in setups with significant laser compression
or implosion, due to macro-particle movement over the simulation run.
The real-world memory usage always depends on the case and the dynamics inside
the simulation and no-movement estimates should be taken with a grain of salt as
an order-of-magnitude estimation.
For a more accurate estimation use estimated compression multipliers.
"""


def check_distributions(simulation_dimension, grid_distribution, gpu_particle_cell_distribution):
    if len(grid_distribution) != simulation_dimension:
        raise ValueError("grid_distribution must have one array for each simulation dimension")
    if len(gpu_particle_cell_distribution) != simulation_dimension:
        raise ValueError("gpu_particle_cell_distribution must have array for each simulation dimension")


def check_extents(simulation_dimension, global_gpu_extent, gpu_cell_extent, gpu_particle_cell_extent):
    if np.shape(gpu_cell_extent) != (simulation_dimension, *global_gpu_extent):
        raise ValueError(
            "grid_distribution and global_gpu_extent are not consistent.\n"
            "\tCheck for each simulation dimension that grid_distribution has as many entries as gpu"
            "rows are specified in global_gpu_extent."
        )

    if np.shape(gpu_particle_cell_extent) != (simulation_dimension, *global_gpu_extent):
        raise ValueError(
            "gpu_particle_cell_distribution and global_gpu_extent are not consistent.\n"
            "\tCheck for each simulation dimension that gpu_particle_cell_distribution"
            "has as many entries as gpu rows are specified in global_gpu_extent."
        )


cell_size = 0.8e-6 / 384.0  # 2.083e-9 m
y0 = 0.5e-6  # position of foil front surface (m)
y1 = 1.5e-6  # position of foil rear surface (m)
L = 10e-9  # pre-plasma scale length (m)
L_cutoff = 4 * L  # pre-plasma length (m)

# number of vacuum cells in front of the target
vacuum_cells = int(np.ceil((y0 - L_cutoff) / cell_size))
# number of target cells over the depth of the target(between surfaces + pre-plasma)
target_cells = int(np.ceil((y1 - y0 + 2 * L_cutoff) / cell_size))

simulation_dimension = 2

# number of cells in the simulation in each direction
global_cell_extent = np.array((256, 1280))[:simulation_dimension]
# number of GPU rows in each direction
global_gpu_extent = np.array((2, 2))[:simulation_dimension]

super_cell_size = np.array((16, 16))

# how many cells each gpu in this row of this dimension handles
#   here we distribute cells evenly, but this is not required, for example [np.array((128,128)), np.array((672, 608))]
grid_distribution = [np.array((128, 128)), np.array((640, 640))]

print(f"grid distribution: {grid_distribution}")

# extent of cells filled with particles for each gpu in this row of this dimension
#   foil surface is orthogonal to picongpu laser propagation direction in positive y-direction
gpu_particle_cell_distribution = [
    grid_distribution[0],
    # number of target cells on first GPU,                    target cells not on first GPU
    np.array(
        (max(grid_distribution[1][0] - vacuum_cells, 0), target_cells - max(grid_distribution[1][0] - vacuum_cells, 0))
    ),
]

print(f"particle cell distribution: {gpu_particle_cell_distribution}")

# debug checks
check_distributions(simulation_dimension, grid_distribution, gpu_particle_cell_distribution)

# get cell extent of each GPU: list of np.array[np.int_], one per simulation dimension, with each array entry being the
#   cell extent of the corresponding gpu in the simulation, indexation by [simulation_dimension, gpu_index[0], gpu_index[1], ...]
gpu_cell_extent = np.meshgrid(*grid_distribution, indexing="ij")

# extent of cells filled with particles of each GPU:
#   same indexation as gpu_cell_extent
gpu_particle_cell_extent = np.meshgrid(*gpu_particle_cell_distribution, indexing="ij")

# debug checks
check_extents(simulation_dimension, global_gpu_extent, gpu_cell_extent, gpu_particle_cell_extent)

mc = MemoryCalculator(simulation_dimension=simulation_dimension, super_cell_size=super_cell_size)

# typical number of particles per cell which is multiplied later for each species and its relative number of particles
N_PPC = 6

# conversion factor to mibi-byte
mibibyte = 1024**2

dimension_name_dict = {0: "x", 1: "y", 2: "z"}

for gpu_index in np.ndindex(tuple(global_gpu_extent)):
    gpu_header_string = "GPU ("
    for dim in range(simulation_dimension):
        gpu_header_string += dimension_name_dict[dim] + f" = {gpu_index[dim]}, "
    print("\n" + gpu_header_string[:-2] + ")")
    print("* Memory requirement:")

    cell_extent = np.empty(simulation_dimension)
    for dim in range(simulation_dimension):
        cell_extent[dim] = gpu_cell_extent[dim][gpu_index]

    # field memory per GPU, see memory.param:fieldTmpNumSlots for number_of_temporary_field_slots
    field_gpu = mc.memory_required_by_cell_fields(cell_extent, number_of_temporary_field_slots=2)
    print(f" + fields: {field_gpu / mibibyte:.2f} MiB")

    # memory for random number generator states
    rng_gpu = mc.memory_required_by_random_number_generator(cell_extent)

    # electron macroparticles per cell
    e_PPC = N_PPC * (
        # H,C,N pre-ionization - higher weighting electrons
        3
        # electrons created from C ionization
        + (6 - 2)
        # electrons created from N ionization
        + (7 - 2)
    )

    particle_cell_extent = np.empty(simulation_dimension)
    for dim in range(simulation_dimension):
        particle_cell_extent[dim] = gpu_particle_cell_extent[dim][gpu_index]

    # particle memory per GPU - only the target area contributes here
    e_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting"],
        particles_per_cell=e_PPC,
    )
    H_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting"],
        particles_per_cell=N_PPC,
    )
    C_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting", "boundElectrons"],
        particles_per_cell=N_PPC,
    )
    N_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting", "boundElectrons"],
        particles_per_cell=N_PPC,
    )

    # memory for calorimeters
    cal_gpu = (
        mc.memory_required_by_calorimeter(number_energy_bins=1024, number_yaw_bins=360, number_pitch_bins=1) * 2
    )  # electrons and protons

    print(" + species:")
    print(f"  - e: {e_gpu / mibibyte:.2f} MiB")
    print(f"  - H: {H_gpu / mibibyte:.2f} MiB")
    print(f"  - C: {C_gpu / mibibyte:.2f} MiB")
    print(f"  - N: {N_gpu / mibibyte:.2f} MiB")
    print(f" + RNG states: {rng_gpu / mibibyte:.2f} MiB")
    print(f" + particle calorimeters: {cal_gpu / mibibyte:.2f} MiB")

    mem_sum = field_gpu + e_gpu + H_gpu + C_gpu + N_gpu + rng_gpu + cal_gpu
    print(f"* Sum of required GPU memory: {mem_sum / mibibyte:.2f} MiB")
